"""Pass #6: media /play chain — verified bugs, failing-first.

1. ``_play_send_with_fallthrough`` send-failure message interpolates the
   literal text "{path}" (missing f-string) while the single-send path
   shows the real path.
2. ``_play_send_with_fallthrough`` on TOTAL failure leaves the
   pre-enqueued first-pick row stranded in the durable queue — the
   single-send path drops it (``_drop_matching_queued``), the
   fall-through path forgot.
3. ``PlaybackEngine.download()`` routes "youtube-music" through the
   generic ``media_download`` instead of the cached ``_youtube_audio``
   path that "youtube" uses (same watch-URL shape, same video ids).
"""

from types import SimpleNamespace
from unittest import mock

import pytest


def _handler():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    return rt


def _chat():
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = "12345"
    return chat


def _item(kind, path, title):
    return {"kind": kind, "path": path, "title": title,
            "artist": "Ayo Maff"}


SC_ITEM = _item("soundcloud",
                "https://soundcloud.com/ayorinde-mafoluku/lifestyle-ya-man",
                "Lifestyle (YA MAN)")

SEARCH_ATTEMPTS = ["soundcloud-search — ok (1)"]


# ── 1: send-failure message shows the real path ───────────────────────


def test_fallthrough_send_failure_shows_real_path_not_literal():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": True, "path": "/tmp/yt-song.mp3",
        "title": "Lifestyle (YA MAN)"}
    rt.gateway.send_file.side_effect = Exception("chat write failed")
    msg = rt._play_send_with_fallthrough(
        engine, SC_ITEM, [], SEARCH_ATTEMPTS, "queued 1:", chat)
    assert "{path}" not in msg, \
        f"message leaks the literal '{{path}}': {msg}"
    assert "/tmp/yt-song.mp3" in msg
    assert "couldn't send" in msg


# ── 2: total failure drops the stranded queue row ─────────────────────


def test_fallthrough_total_failure_drops_stranded_queue_row():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": False, "reason": "This video is DRM protected"}
    # The durable queue holds the pre-enqueued first pick (index 0).
    engine.queue.return_value = [
        {"index": 0, "kind": "soundcloud",
         "path": SC_ITEM["path"], "title": SC_ITEM["title"]}]
    removed = []
    engine.remove.side_effect = lambda idx: removed.append(idx)
    msg = rt._play_send_with_fallthrough(
        engine, SC_ITEM, [], SEARCH_ATTEMPTS, "queued 1:", chat)
    assert "couldn't download" in msg
    assert removed == [0], \
        f"stranded queue row was not dropped on total failure: {removed}"


# ── 3: youtube-music uses the cached audio path ───────────────────────


def _download_engine():
    from nomorals.media.playback import PlaybackEngine

    ctx = mock.MagicMock()
    tools = mock.MagicMock()
    media_download_calls = []
    tools.call.side_effect = lambda name, **kw: (
        media_download_calls.append((name, kw)),
        SimpleNamespace(ok=True, value={"path": "/tmp/dl.mp3"}))[1]
    ctx.tools = tools
    engine = PlaybackEngine.__new__(PlaybackEngine)
    engine.context = ctx
    engine._youtube_audio = mock.Mock(return_value="/tmp/cached.mp3")
    return engine, media_download_calls


def test_youtube_music_download_uses_cached_audio_path():
    engine, media_download_calls = _download_engine()
    res = engine.download(_item(
        "youtube-music", "https://www.youtube.com/watch?v=abcdefghijk",
        "Lifestyle (YA MAN)"))
    assert res["ok"] is True
    assert res["path"] == "/tmp/cached.mp3"
    assert engine._youtube_audio.call_count == 1
    assert media_download_calls == [], \
        "youtube-music bypassed the _youtube_audio cache"
