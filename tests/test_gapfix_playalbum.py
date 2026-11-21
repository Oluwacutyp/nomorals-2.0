"""Tests for `/play L <artist>` — a random album, every track sent.

Rules under test:
- (a) Album metadata comes from the identification-only layer
  (iTunes album search + lookup, Deezer fallback): real albums are
  named, singles are marked single and only picked when the artist
  has no full album, namesakes are filtered out, duplicates collapse.
- (b) The pick is random per invocation (runtime-seeded, never a
  fixed seed): 30 unseeded picks over 5 albums land on at least 3
  distinct albums; a seeded rng is deterministic (tests only).
- (c) Honest failure: no albums → named; album with no track list →
  named; a track that fails → the rest still send and the summary
  names the missing track and its reason. No fake album, no silent
  single-track fallback, no silent truncation.
- (d) The send path is the single-/play fall-through per track
  (resolve → candidate sweep → `_play_send_with_fallthrough` with a
  (title, artist) match gate); preview-only tracks are never sent.
- (e) Chat: one background job behind one ack; the ack is edited to
  the album summary. CLI: queue the whole album, then play.
"""

import random
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.media.play_album import (
    Album,
    AlbumTrack,
    album_tracks,
    find_album_for_artist,
    format_album_summary,
    pick_random_album,
    search_albums,
)
from nomorals.media.resolver import ResolvedAudio

ITUNES_ALBUMS = {
    "resultCount": 5,
    "results": [
        {"wrapperType": "collection", "collectionType": "Album",
         "collectionId": 11, "collectionName": "Alpha",
         "artistName": "Test Artist", "trackCount": 10},
        {"wrapperType": "collection", "collectionType": "Album",
         "collectionId": 22, "collectionName": "Beta - Single",
         "artistName": "Test Artist", "trackCount": 1},
        {"wrapperType": "collection", "collectionType": "Album",
         "collectionId": 33, "collectionName": "Gamma",
         "artistName": "Test Artist", "trackCount": 12},
        {"wrapperType": "collection", "collectionType": "Album",
         "collectionId": 11, "collectionName": "Alpha",
         "artistName": "Test Artist", "trackCount": 10},
        {"wrapperType": "collection", "collectionType": "Album",
         "collectionId": 55, "collectionName": "Other Band Album",
         "artistName": "Other Band", "trackCount": 9},
    ],
}

ITUNES_TRACKS = {
    "resultCount": 4,
    "results": [
        {"wrapperType": "collection", "collectionId": 11,
         "collectionName": "Alpha"},
        {"wrapperType": "track", "trackId": 1,
         "trackName": "First Song", "trackNumber": 1},
        {"wrapperType": "track", "trackId": 2,
         "trackName": "Second Song", "trackNumber": 2},
        {"wrapperType": "track", "trackId": 3,
         "trackName": "Third Song", "trackNumber": 3},
    ],
}

DEEZER_ALBUMS = {
    "data": [
        {"id": 101, "title": "Delta",
         "artist": {"name": "Test Artist"}, "nb_tracks": 8},
        {"id": 102, "title": "FIFA Compilation",
         "artist": {"name": "Various Artists"}, "nb_tracks": 20},
    ],
}

DEEZER_TRACKS = {
    "data": [
        {"id": 1, "title": "Delta One", "track_position": 1},
        {"id": 2, "title": "Delta Two", "track_position": 2},
    ],
}

EMPTY_ITUNES = {"resultCount": 0, "results": []}
EMPTY_DEEZER = {"data": []}


def _canned_fetcher(*, itunes_albums=ITUNES_ALBUMS,
                    itunes_tracks=ITUNES_TRACKS,
                    deezer_albums=DEEZER_ALBUMS,
                    deezer_tracks=DEEZER_TRACKS):
    """A fetcher serving canned payloads by URL; records every call."""
    import json

    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        if "itunes.apple.com/search" in url:
            return json.dumps(itunes_albums)
        if "itunes.apple.com/lookup" in url:
            return json.dumps(itunes_tracks)
        if "api.deezer.com/search/album" in url:
            return json.dumps(deezer_albums)
        if "api.deezer.com/album/" in url:
            return json.dumps(deezer_tracks)
        raise AssertionError(f"unexpected fetch: {url}")

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def _failing_fetcher(url: str) -> str:
    raise ConnectionError("network down")


# ── module: album search ──────────────────────────────────────────────

def test_itunes_album_search_parses_filters_and_dedupes():
    fetch = _canned_fetcher()
    albums = search_albums("Test Artist", fetcher=fetch)
    assert [a.title for a in albums] == ["Alpha", "Gamma", "Beta - Single"]
    assert [a.kind for a in albums] == ["album", "album", "single"]
    assert [a.source for a in albums] == ["itunes"] * 3
    assert albums[0].source_id == "11"
    assert albums[0].track_count == 10
    # Deezer never called: iTunes answered.
    assert not any("deezer.com" in u for u in fetch.calls)


def test_itunes_lookup_orders_tracks():
    album = Album(title="Alpha", artist="Test Artist", source="itunes",
                  source_id="11", kind="album", track_count=10)
    tracks = album_tracks(album, fetcher=_canned_fetcher())
    assert [t.title for t in tracks] == [
        "First Song", "Second Song", "Third Song"]
    assert [t.number for t in tracks] == [1, 2, 3]


def test_deezer_fallback_when_itunes_empty():
    fetch = _canned_fetcher(itunes_albums=EMPTY_ITUNES)
    albums = search_albums("Test Artist", fetcher=fetch)
    assert [a.title for a in albums] == ["Delta"]
    assert albums[0].source == "deezer"
    assert albums[0].source_id == "101"
    tracks = album_tracks(albums[0], fetcher=fetch)
    assert [t.title for t in tracks] == ["Delta One", "Delta Two"]


def test_search_albums_empty_when_both_fail():
    assert search_albums("Test Artist", fetcher=_failing_fetcher) == []
    assert search_albums("", fetcher=_canned_fetcher()) == []


def test_album_tracks_empty_without_source_id():
    assert album_tracks(
        Album(title="X", artist="Y", source="itunes", source_id=""),
        fetcher=_canned_fetcher()) == []


# ── module: random pick ───────────────────────────────────────────────

def test_pick_random_album_prefers_full_albums():
    fetch = _canned_fetcher()
    albums = search_albums("Test Artist", fetcher=fetch)
    for seed in range(100):
        picked = pick_random_album(albums, rng=random.Random(seed))
        assert picked is not None
        assert picked.kind == "album"


def test_pick_random_album_singles_only_when_nothing_else():
    singles = [Album(title="One - Single", artist="A", source="itunes",
                     source_id="1", kind="single"),
               Album(title="Two - Single", artist="A", source="itunes",
                     source_id="2", kind="single")]
    picked = pick_random_album(singles, rng=random.Random(7))
    assert picked is not None
    assert picked.kind == "single"


def test_pick_random_album_is_random_not_fixed():
    albums = [Album(title=f"Album {i}", artist="A", source="itunes",
                    source_id=str(i), kind="album") for i in range(5)]
    seen = {pick_random_album(albums).title for _ in range(30)}
    assert len(seen) >= 3


def test_pick_random_album_empty():
    assert pick_random_album([]) is None


def test_pick_random_album_seeded_is_deterministic():
    albums = [Album(title=f"Album {i}", artist="A", source="itunes",
                    source_id=str(i), kind="album") for i in range(5)]
    first = pick_random_album(albums, rng=random.Random(42))
    second = pick_random_album(albums, rng=random.Random(42))
    assert first == second


# ── module: find + summary ────────────────────────────────────────────

def test_find_album_happy_path():
    result = find_album_for_artist("Test Artist",
                                   fetcher=_canned_fetcher(),
                                   rng=random.Random(1))
    assert result.error == ""
    assert result.album is not None
    assert result.album.title in ("Alpha", "Gamma")
    assert [t.title for t in result.tracks] == [
        "First Song", "Second Song", "Third Song"]


def test_find_album_no_albums_names_the_artist():
    result = find_album_for_artist(
        "Nobody Real", fetcher=_canned_fetcher(
            itunes_albums=EMPTY_ITUNES, deezer_albums=EMPTY_DEEZER))
    assert result.album is None
    assert "Nobody Real" in result.error
    assert "no albums found" in result.error


def test_find_album_no_tracks_is_honest():
    result = find_album_for_artist(
        "Test Artist",
        fetcher=_canned_fetcher(itunes_tracks={"resultCount": 0,
                                               "results": []}),
        rng=random.Random(1))
    assert result.album is not None
    assert result.tracks == []
    assert "no track list" in result.error


def test_find_album_blank_artist_is_usage():
    result = find_album_for_artist("   ")
    assert result.album is None
    assert "usage" in result.error


def test_format_album_summary_all_delivered():
    album = Album(title="Alpha", artist="Test Artist")
    text = format_album_summary(album, ["A", "B", "C"], [])
    assert "Alpha — Test Artist" in text
    assert "3 of 3 tracks sent" in text
    assert "missing" not in text


def test_format_album_summary_names_missing():
    album = Album(title="Alpha", artist="Test Artist")
    text = format_album_summary(
        album, ["First Song", "Second Song"],
        [("Third Song", "couldn't find 'Third Song'")])
    assert "2 of 3 tracks sent" in text
    assert "Third Song" in text
    assert "couldn't find" in text


def test_format_album_summary_none_delivered():
    album = Album(title="Alpha", artist="Test Artist")
    text = format_album_summary(
        album, [], [("First Song", "couldn't find it")])
    assert "0 of 1 track sent" in text
    assert "First Song" in text


# ── chat: harness (fall-through pattern) ──────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    from nomorals.media import download_failures as _df

    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    _df.clear_failures()
    yield
    _df.clear_failures()


class _FakeGateway:
    """Records sends/files/edits; plays the adapter for edit verbs."""

    def __init__(self):
        self.sent = []
        self.files = []
        self.edits = []
        self._n = 0

    def send(self, platform, chat, text, **kw):
        self._n += 1
        self.sent.append((platform, chat, text))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id=f"m{self._n}")

    def send_file(self, platform, target, path, caption="", kind=None):
        self.files.append((platform, target, path, caption))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id="f1")

    def _adapter_for(self, platform):
        return self

    def edit_message(self, chat, message_id, text, *, buttons=None):
        self.edits.append((str(message_id), text))
        return {"ok": True}

    def edit_progress(self, chat, message_id, text):
        self.edits.append((str(message_id), text))
        return {"ok": True}


class _FakeEngine:
    """PlaybackEngine stand-in: enqueue only, no playback, no network."""

    def __init__(self):
        self.queued = []

    def _enqueue(self, path, kind, title, artist="", duration=0.0):
        # The real engine's rows carry their queue position as
        # "index" — _queued_index_of (drop-matching) reads it.
        item = {"index": len(self.queued), "path": path, "kind": kind,
                "title": title, "artist": artist, "duration": duration}
        self.queued.append(item)
        return dict(item)

    def queue(self):
        return list(self.queued)

    def remove(self, idx):
        return self.queued.pop(idx)["title"]

    def play(self, index=0):
        if not self.queued:
            return {"status": "empty"}
        return {"status": "playing",
                "current": self.queued[index]["title"],
                "backend": "fake"}


def _handler():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = _FakeGateway()
    return rt


def _chat(chat_id="12345"):
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = chat_id
    chat.key = f"telegram:{chat_id}"
    return chat


def _collect_album_job(chat_key, title, timeout=10.0):
    from nomorals.core.jobs import shared_registry

    snaps = shared_registry().list(
        labels={"owner": "play", "chat": chat_key, "title": title})
    assert snaps, f"no play job for {chat_key}/{title}"
    job = shared_registry().collect(snaps[-1]["id"], timeout=timeout)
    assert job is not None and job.done(), "album job did not finish"
    return job


def _album_result():
    return SimpleNamespace(
        album=Album(title="Alpha", artist="Test Artist", source="itunes",
                    source_id="11", kind="album", track_count=3),
        tracks=[AlbumTrack(number=1, title="First Song"),
                AlbumTrack(number=2, title="Second Song"),
                AlbumTrack(number=3, title="Third Song")],
        error="")


def test_play_L_usage_without_artist():
    rt = _handler()
    assert "usage" in rt._control_play("L", chat_key="telegram:12345")
    assert "usage" in rt._control_play("L   ", chat_key="telegram:12345")
    assert rt.gateway.sent == []


def test_play_L_album_sends_every_track_in_order(monkeypatch):
    rt = _handler()
    rt._ref_from_key = lambda key: _chat()
    engine = _FakeEngine()
    monkeypatch.setattr("nomorals.media.playback.PlaybackEngine",
                        lambda context: engine)
    monkeypatch.setattr(
        "nomorals.media.play_album.find_album_for_artist",
        lambda artist, **kw: _album_result())
    calls: list[str] = []

    def _send_track(_engine, artist, title, chat):
        calls.append(title)
        if title == "Third Song":
            return "couldn't find 'Third Song'", False
        return f"🎵 {title} — {artist}", True

    rt._play_album_track = _send_track
    out = rt._control_play("L Test Artist", chat_key="telegram:12345")
    assert out == ""  # the ack already posted
    assert len(rt.gateway.sent) == 1
    assert "Test Artist" in rt.gateway.sent[0][2]
    _collect_album_job("telegram:12345", "album: Test Artist")
    assert calls == ["First Song", "Second Song", "Third Song"]
    assert rt.gateway.edits, "ack was never edited to the summary"
    final = rt.gateway.edits[-1][1]
    assert "Alpha — Test Artist" in final
    assert "2 of 3 tracks sent" in final
    assert "Third Song" in final


def test_play_L_no_album_is_honest_and_sends_nothing(monkeypatch):
    rt = _handler()
    rt._ref_from_key = lambda key: _chat()
    engine = _FakeEngine()
    monkeypatch.setattr("nomorals.media.playback.PlaybackEngine",
                        lambda context: engine)
    monkeypatch.setattr(
        "nomorals.media.play_album.find_album_for_artist",
        lambda artist, **kw: SimpleNamespace(
            album=None, tracks=[], error="no albums found for 'Nobody'"))
    out = rt._control_play("L Nobody", chat_key="telegram:12345")
    assert out == ""
    _collect_album_job("telegram:12345", "album: Nobody")
    assert rt.gateway.files == []
    final = rt.gateway.edits[-1][1]
    assert "no albums found" in final
    assert "Nobody" in final


def test_play_L_lowercase_verb_works(monkeypatch):
    rt = _handler()
    rt._ref_from_key = lambda key: _chat()
    engine = _FakeEngine()
    monkeypatch.setattr("nomorals.media.playback.PlaybackEngine",
                        lambda context: engine)
    monkeypatch.setattr(
        "nomorals.media.play_album.find_album_for_artist",
        lambda artist, **kw: _album_result())
    rt._play_album_track = lambda e, a, t, c: (f"🎵 {t}", True)
    out = rt._control_play("l Test Artist", chat_key="telegram:12345")
    assert out == ""
    _collect_album_job("telegram:12345", "album: Test Artist")
    assert "3 of 3 tracks sent" in rt.gateway.edits[-1][1]


def test_play_album_track_sends_via_fallthrough(monkeypatch):
    rt = _handler()
    engine = _FakeEngine()
    chat = _chat()
    resolution = ResolvedAudio(
        ok=True, path_or_url="https://mp3.pm/track/1", title="First Song",
        artist="Test Artist", kind="url", source_name="mp3.pm",
        attempts=["mp3pm-search — ok (1)"], downloadable=True)
    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver.resolve",
        lambda self, query: resolution)
    candidate = ResolvedAudio(
        ok=True, path_or_url="https://mp3.pm/track/1", title="First Song",
        artist="Test Artist", kind="url", downloadable=True)
    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver.search_candidates_detailed",
        lambda self, query, limit=8: ([candidate], ["mp3pm — ok (1)"]))
    seen: dict[str, Any] = {}

    def _fallthrough(_engine, first_item, fallbacks, attempts, _chat,
                     match=None, query=""):
        seen["first"] = first_item
        seen["match"] = match
        seen["fallbacks"] = fallbacks
        return "🎵 First Song — Test Artist", True

    rt._play_send_with_fallthrough = _fallthrough
    text, sent = rt._play_album_track(engine, "Test Artist", "First Song",
                                      chat)
    assert sent is True
    assert "First Song" in text
    assert seen["match"] == ("First Song", "Test Artist")
    assert seen["first"]["title"] == "First Song"
    assert len(engine.queued) == 1


def test_play_album_track_unresolved_never_sends(monkeypatch):
    rt = _handler()
    engine = _FakeEngine()
    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver.resolve",
        lambda self, query: ResolvedAudio(ok=False, attempts=["nope"]))
    rt._play_send_with_fallthrough = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("fall-through must not run"))
    text, sent = rt._play_album_track(engine, "Test Artist", "Ghost Song",
                                      _chat())
    assert sent is False
    assert "Ghost Song" in text
    assert engine.queued == []


def test_play_album_track_preview_only_is_a_miss_not_a_send(monkeypatch):
    rt = _handler()
    engine = _FakeEngine()
    resolution = ResolvedAudio(
        ok=True, path_or_url="https://itunes.apple.com/x",
        title="Preview Song", artist="Test Artist", kind="url",
        attempts=["itunes — ok (1)"], downloadable=False,
        preview_only=True)
    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver.resolve",
        lambda self, query: resolution)
    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver.search_candidates_detailed",
        lambda self, query, limit=8: ([], ["itunes — ok (0)"]))
    rt._play_send_with_fallthrough = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("fall-through must not run for a preview"))
    text, sent = rt._play_album_track(engine, "Test Artist", "Preview Song",
                                      _chat())
    assert sent is False
    assert "preview" in text
    assert engine.queued == []


def test_play_L_cli_queues_and_plays(monkeypatch):
    rt = _handler()
    engine = _FakeEngine()
    monkeypatch.setattr("nomorals.media.playback.PlaybackEngine",
                        lambda context: engine)
    monkeypatch.setattr(
        "nomorals.media.play_album.find_album_for_artist",
        lambda artist, **kw: _album_result())
    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver.resolve",
        lambda self, query: ResolvedAudio(
            ok=True, path_or_url="https://mp3.pm/track/1",
            title=query, artist="Test Artist", kind="url"))
    out = rt._control_play("L Test Artist")
    assert "Alpha — Test Artist" in out
    assert "3 of 3 tracks sent" in out
    assert "playing" in out
    assert [q["title"] for q in engine.queued] == [
        "Test Artist First Song", "Test Artist Second Song",
        "Test Artist Third Song"]
