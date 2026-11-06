"""Tests for /play download fall-through (owner screenshot: /play lifestyle (YA MAN)).

The bug: the resolver chain stopped at the first search-ok source
(SoundCloud), then the chat path downloaded that ONE candidate. When the
SoundCloud download failed (this track is DRM-protected — yt-dlp itself
reports "This video is DRM protected" for track 2361175013), the reply
was "couldn't send the audio: ..." and only 5 of 16 sources had been
tried. Download failure must fall through to the NEXT source until one
both finds AND downloads, or all sources are exhausted.

Rules under test:
- (a) source A search-ok + download-fail -> chain tries source B, sends B.
- (b) all sources download-fail -> honest "couldn't download", never
      "couldn't send the audio".  The full tried-list is journal-only
      since the 2026-11-05 presentation overhaul — it is asserted in
      the LOG (caplog), never in the chat text.
- (c) the YouTube-cookies hint appears ONLY when YouTube was actually
      attempted AND bot-blocked; it is SHORT (points at /cookies),
      never the yt-dlp command dump.
- (d) presentation law: success returns the delivered item's own
      line; no "sources tried", no "sent … to this chat" receipts.
"""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media.cookies import BOT_COOKIE_HELP
from nomorals.media.resolver import ResolvedAudio, SourceResolver

BOTCHECK_MSG = (
    "ERROR: [youtube] abc: Sign in to confirm you're not a bot. "
    "Use --cookies-from-browser or --cookies for the authentication."
)

_RT_LOGGER = "nomorals.agents.partner.runtime_media"


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

    def send_file(self, platform, target, path, caption=""):
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


def _item(kind, path, title):
    return {"kind": kind, "path": path, "title": title, "artist": "Ayo Maff"}


def _collect_play_job(chat_key, title, timeout=10.0):
    from nomorals.core.jobs import shared_registry

    snaps = shared_registry().list(
        labels={"owner": "play", "chat": chat_key, "title": title})
    assert snaps, f"no play job for {chat_key}/{title}"
    job = shared_registry().collect(snaps[-1]["id"], timeout=timeout)
    assert job is not None and job.done(), \
        f"job never finished: {job.status if job else None}"
    return job


SC_ITEM = _item("soundcloud",
                "https://soundcloud.com/ayorinde-mafoluku/lifestyle-ya-man",
                "Lifestyle (YA MAN)")
YT_ITEM = _item("youtube", "https://www.youtube.com/watch?v=abcdefghijk",
                "Lifestyle (YA MAN)")
BC_ITEM = _item("bandcamp", "https://example.bandcamp.com/track/x",
                "Lifestyle (YA MAN)")

SEARCH_ATTEMPTS = [
    "workspace-scan — failed",
    "netnaija-search — no results",
    "notjustok-search — no results",
    "tooxclusive-search — no results",
    "soundcloud-search — ok (1)",
    "bandcamp-search — ok (1)",
    "youtube-search — ok (1)",
    "youtube-music-search — no results",
    "archiveorg-search — no results",
    "musicbrainz-search — no results",
    "openverse-search — no results",
    "wikimedia-search — no results",
    "itunes-search — no results",
    "deezer-search — no results",
]


# ── (a) fall-through: A fails download, B succeeds ────────────────────


def test_fallthrough_soundcloud_fail_tries_youtube_and_sends():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": True, "path": "/tmp/yt-song.mp3", "title": "Lifestyle (YA MAN)"},
    ]
    out, sent = rt._play_send_with_fallthrough(
        engine, SC_ITEM, [YT_ITEM, BC_ITEM], SEARCH_ATTEMPTS, chat)
    assert engine.download.call_count == 2
    assert engine.download.call_args_list[0][0][0]["kind"] == "soundcloud"
    assert engine.download.call_args_list[1][0][0]["kind"] == "youtube"
    assert sent is True
    assert len(rt.gateway.files) == 1
    assert "/tmp/yt-song.mp3" in rt.gateway.files[0]
    # The delivered item's own line — no receipt around it.
    assert out == "🎵 Lifestyle (YA MAN) — Ayo Maff"
    assert "to this chat" not in out
    assert "sources tried" not in out
    assert "couldn't download" not in out
    assert "couldn't send" not in out


def test_fallthrough_tries_all_candidates_in_order():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "DRM protected"},
        {"ok": False, "reason": "Bandcamp download failed: 404"},
        {"ok": True, "path": "/tmp/arc.mp3", "title": "Lifestyle (YA MAN)"},
    ]
    arc = _item("archiveorg", "https://archive.org/x.mp3",
                "Lifestyle (YA MAN)")
    out, sent = rt._play_send_with_fallthrough(
        engine, SC_ITEM, [BC_ITEM, arc], SEARCH_ATTEMPTS, chat)
    assert engine.download.call_count == 3
    assert sent is True
    assert out.startswith("🎵")
    assert len(rt.gateway.files) == 1


# ── (b) all fail: honest "couldn't download"; trail in the journal ────


def test_all_download_fail_honest_message_no_send_wording(caplog):
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": False, "reason": "Bandcamp download failed: 404"},
        {"ok": False, "reason": "YouTube download failed: video unavailable"},
    ]
    with caplog.at_level(logging.INFO, logger=_RT_LOGGER):
        out, sent = rt._play_send_with_fallthrough(
            engine, SC_ITEM, [BC_ITEM, YT_ITEM], SEARCH_ATTEMPTS, chat)
    assert sent is False
    assert "couldn't download" in out
    assert "couldn't send the audio" not in out
    assert "couldn't send it" not in out
    # The chat text carries NO tried-list (presentation law)…
    assert "sources tried" not in out
    assert "soundcloud-search" not in out
    # …the full trail is in the journal instead: every source
    # actually attempted, with its download-stage outcome.
    journal = "\n".join(r.getMessage() for r in caplog.records
                        if "play tried" in r.getMessage())
    assert journal, "tried trail never reached the journal"
    for label in ("soundcloud-search", "bandcamp-search", "youtube-search",
                  "netnaija-search", "archiveorg-search"):
        assert label in journal
    assert "download blocked" in journal
    assert rt.gateway.files == []


def test_single_item_send_wording_is_download_not_send():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.return_value = {"ok": False, "reason": "no yt-dlp"}
    out = rt._play_send_in_chat(engine, {"title": "X"}, chat)
    assert "couldn't download" in out
    assert "couldn't send the audio" not in out
    assert rt.gateway.files == []


def test_fallthrough_skips_non_downloadable():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    boom = dict(_item("boomplay", "https://boomplay.com/x", "Lifestyle"),
                downloadable=False)
    engine.download.return_value = {
        "ok": True, "path": "/tmp/sc.mp3", "title": "Lifestyle (YA MAN)"}
    # First item downloads fine; the non-downloadable fallback is never
    # attempted — it is only recorded.
    out, sent = rt._play_send_with_fallthrough(
        engine, SC_ITEM, [boom], SEARCH_ATTEMPTS, chat)
    assert engine.download.call_count == 1
    assert sent is True


def test_fallthrough_records_non_downloadable_without_attempt(caplog):
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.return_value = {"ok": False, "reason": "DRM protected"}
    boom = dict(_item("boomplay", "https://boomplay.com/x", "Lifestyle"),
                downloadable=False)
    with caplog.at_level(logging.INFO, logger=_RT_LOGGER):
        out, sent = rt._play_send_with_fallthrough(
            engine, SC_ITEM, [boom], SEARCH_ATTEMPTS, chat)
    # Only the downloadable candidate was attempted...
    assert engine.download.call_count == 1
    assert sent is False
    assert "couldn't download" in out
    # ...and Boomplay is accounted for in the journal, by name.
    journal = "\n".join(r.getMessage() for r in caplog.records)
    assert "boomplay" in journal.lower()
    assert "not downloadable" in journal


# ── (c) cookies hint: only when YouTube attempted AND bot-blocked ─────


def test_cookies_hint_when_youtube_bot_blocked():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": False, "reason": f"YouTube download failed: {BOTCHECK_MSG}"},
    ]
    out, sent = rt._play_send_with_fallthrough(
        engine, SC_ITEM, [YT_ITEM], SEARCH_ATTEMPTS, chat)
    assert sent is False
    assert "couldn't download" in out
    assert "/cookies" in out
    # SHORT hint only — never the yt-dlp command dump.
    assert BOT_COOKIE_HELP not in out
    assert "--cookies-from-browser chrome" not in out


def test_no_cookies_hint_when_youtube_fails_generic():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": False, "reason": "YouTube download failed: video unavailable"},
    ]
    out, _sent = rt._play_send_with_fallthrough(
        engine, SC_ITEM, [YT_ITEM], SEARCH_ATTEMPTS, chat)
    assert "/cookies" not in out
    assert BOT_COOKIE_HELP not in out


def test_no_cookies_hint_when_youtube_never_attempted():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": False, "reason": "Bandcamp download failed: 404"},
    ]
    attempts = [a for a in SEARCH_ATTEMPTS
                if not a.startswith("youtube-search")]
    out, _sent = rt._play_send_with_fallthrough(
        engine, SC_ITEM, [BC_ITEM], attempts, chat)
    assert "/cookies" not in out
    assert BOT_COOKIE_HELP not in out


def test_skip_duplicate_youtube_after_fallback_attempt(caplog):
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    # SoundCloud's own download already tried the YouTube fallback and
    # hit the bot check — do NOT attempt the YouTube candidate again.
    engine.download.return_value = {
        "ok": False,
        "reason": f"SoundCloud blocked the download (DRM). {BOTCHECK_MSG}",
        "youtube_fallback_attempted": True,
    }
    with caplog.at_level(logging.INFO, logger=_RT_LOGGER):
        out, sent = rt._play_send_with_fallthrough(
            engine, SC_ITEM, [YT_ITEM], SEARCH_ATTEMPTS, chat)
    assert engine.download.call_count == 1
    assert sent is False
    # The skip is accounted for in the journal…
    journal = "\n".join(r.getMessage() for r in caplog.records)
    assert "already attempted" in journal
    assert "/cookies" in out  # YouTube WAS attempted (via fallback) + blocked


# ── resolver: detailed per-source attempts ─────────────────────────────


def test_search_candidates_detailed_reports_all_sources():
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    with mock.patch.object(
            r, "_netnaija_search_many",
            return_value=[ResolvedAudio(ok=True, path_or_url="https://n/x",
                                        title="Lifestyle", kind="netnaija",
                                        source_name="NetNaija search")]), \
            mock.patch.object(r, "_soundcloud_search_many",
                              side_effect=RuntimeError("boom")), \
            mock.patch.object(r, "_youtube_search_many", return_value=[]), \
            mock.patch.object(r, "_youtube_music_search_many",
                              return_value=[]), \
            mock.patch.object(r, "_bandcamp_search_many", return_value=[]), \
            mock.patch.object(r, "_notjustok_search_many", return_value=[]), \
            mock.patch.object(r, "_tooxclusive_search_many", return_value=[]), \
            mock.patch.object(r, "_archiveorg_search_many", return_value=[]), \
            mock.patch.object(r, "_musicbrainz_search_many", return_value=[]), \
            mock.patch.object(r, "_openverse_search_many", return_value=[]), \
            mock.patch.object(r, "_wikimedia_search_many", return_value=[]), \
            mock.patch.object(r, "_itunes_search_many", return_value=[]), \
            mock.patch.object(r, "_deezer_search_many", return_value=[]):
        cands, attempts = r.search_candidates_detailed("lifestyle", limit=8)
    assert [c.title for c in cands] == ["Lifestyle"]
    joined = " | ".join(attempts)
    for name in ("netnaija-search", "notjustok-search",
                 "tooxclusive-search", "soundcloud-search",
                 "bandcamp-search", "youtube-search",
                 "youtube-music-search",
                 "archiveorg-search", "musicbrainz-search",
                 "openverse-search", "wikimedia-search",
                 "itunes-search", "deezer-search"):
        assert name in joined, joined
    assert "netnaija-search — ok" in joined
    assert "soundcloud-search — failed" in joined
    assert "youtube-search — no results" in joined


def test_search_candidates_still_returns_candidates_only():
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    with mock.patch.object(
            r, "_netnaija_search_many",
            return_value=[ResolvedAudio(ok=True, path_or_url="https://n/x",
                                        title="Lifestyle", kind="netnaija",
                                        source_name="NetNaija search")]), \
            mock.patch.object(r, "_soundcloud_search_many", return_value=[]), \
            mock.patch.object(r, "_youtube_search_many", return_value=[]), \
            mock.patch.object(r, "_youtube_music_search_many",
                              return_value=[]), \
            mock.patch.object(r, "_bandcamp_search_many", return_value=[]), \
            mock.patch.object(r, "_notjustok_search_many", return_value=[]), \
            mock.patch.object(r, "_tooxclusive_search_many", return_value=[]), \
            mock.patch.object(r, "_archiveorg_search_many", return_value=[]), \
            mock.patch.object(r, "_musicbrainz_search_many", return_value=[]), \
            mock.patch.object(r, "_openverse_search_many", return_value=[]), \
            mock.patch.object(r, "_wikimedia_search_many", return_value=[]), \
            mock.patch.object(r, "_itunes_search_many", return_value=[]), \
            mock.patch.object(r, "_deezer_search_many", return_value=[]):
        out = r.search_candidates("lifestyle", limit=8)
    assert [c.title for c in out] == ["Lifestyle"]


# ── end-to-end through _control_play (chat path) ───────────────────────


def test_control_play_chat_falls_through_to_youtube():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = _FakeGateway()
    chat = _chat()
    rt._ref_from_key = lambda key: chat

    sc_res = ResolvedAudio(
        ok=True, path_or_url=SC_ITEM["path"], title="Lifestyle (YA MAN)",
        artist="Ayo Maff", kind="soundcloud",
        source_name="SoundCloud search",
        attempts=["workspace-scan — failed",
                  "netnaija-search — failed",
                  "notjustok-search — failed",
                  "tooxclusive-search — failed",
                  "soundcloud-search — ok (SoundCloud search)"])
    sc_cand = ResolvedAudio(
        ok=True, path_or_url=SC_ITEM["path"], title="Lifestyle (YA MAN)",
        artist="Ayo Maff", kind="soundcloud",
        source_name="SoundCloud search")
    yt_cand = ResolvedAudio(
        ok=True, path_or_url=YT_ITEM["path"], title="Lifestyle (YA MAN)",
        artist="Ayo Maff", kind="youtube",
        source_name="YouTube search (yt-dlp)")
    detailed_attempts = [
        "netnaija-search — no results",
        "notjustok-search — no results",
        "tooxclusive-search — no results",
        "soundcloud-search — ok (1)",
        "bandcamp-search — no results",
        "youtube-search — ok (1)",
        "youtube-music-search — no results",
        "archiveorg-search — no results",
        "musicbrainz-search — no results",
        "openverse-search — no results",
        "wikimedia-search — no results",
        "itunes-search — no results",
        "deezer-search — no results",
    ]

    engine = mock.Mock()
    engine._enqueue.side_effect = lambda p, k, t, **kw: {
        "kind": k, "path": p, "title": t, "artist": kw.get("artist", "")}
    engine.queue.return_value = [{"title": "Lifestyle (YA MAN)"}]
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": True, "path": "/tmp/yt-song.mp3",
         "title": "Lifestyle (YA MAN)"},
    ]
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = sc_res
        SR.return_value.search_candidates_detailed.return_value = (
            [sc_cand, yt_cand], detailed_attempts)
        out = rt._control_play("lifestyle (YA MAN)", "chat-key")
    # The handler returns "" — the ack already posted as a message.
    assert out == ""
    job = _collect_play_job("chat-key", "Lifestyle (YA MAN)")
    assert job.result["sent"] is True
    assert engine.download.call_count == 2
    assert len(rt.gateway.files) == 1
    assert "/tmp/yt-song.mp3" in rt.gateway.files[0]
    # The whole surface: one ack send, the file, ack edits.  No
    # tried-list, no receipt, no queue counts anywhere.
    surface = [t for _p, _c, t in rt.gateway.sent]
    surface += [t for _m, t in rt.gateway.edits]
    surface += [c for _p, _t, _path, c in rt.gateway.files]
    assert surface[0].startswith("⏳")
    for text in surface:
        assert "sources tried" not in text
        assert "to this chat" not in text
        assert "[job " not in text
    # The final state is the delivered item's line, edited in.
    assert rt.gateway.edits[-1][1] == "🎵 Lifestyle (YA MAN) — Ayo Maff"
