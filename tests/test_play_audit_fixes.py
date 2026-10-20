"""Audit fixes for the WHOLE /play path (beyond the original fall-through).

Each bug found by auditing the full chain — source chain, queue,
search -> download -> send handoffs, error messages at every stage,
yt-dlp/cookies handling, SoundCloud/YouTube failures:

- BUG #2  resolver chain stops on a NON-downloadable "ok"
  (Boomplay/MusicBrainz) — later downloadable sources never tried.
- BUG #3  fall-through queue mismatch: queue keeps the failed first
  pick while a different file was sent.
- BUG #4  failed single-send leaves the broken item in the durable queue.
- BUG #5  /play pick consumes the pick-list BEFORE download, against
  its own "consumed on a successful pick" contract — a failed pick
  loses the whole list.
- BUG #6  pick-list shows no source and no 30s-preview warning.
- BUG #7  candidate sweep returns non-downloadable (MusicBrainz)
  entries as pickable/downloadable.
- BUG #8  single-send path never appends the cookies hint on a
  YouTube bot-check.
- BUG #9  forced-YouTube exception paths lack the cookies hint.
- BUG #10 "couldn't find" chat path renders the FULL yt-dlp command
  dump instead of the short /cookies hint.
- BUG #11 NL "play X" drops chat_key (no file sent) and bypasses
  the music feature gate.
- BUG #12 bare/invalid seek/volume/remove -> "play error: <raw
  exception>" (e.g. "list index out of range").
- BUG #13 /cookies reports "found" for a file with no valid
  (non-expired) YouTube cookies.
- BUG #14 forced-YouTube CLI double-enqueues (once in _control_play,
  once inside play_youtube).
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media.cookies import BOT_COOKIE_HELP
from nomorals.media.picklist import PickCache, format_picklist
from nomorals.media.resolver import ResolvedAudio, SourceResolver


def _ctx():
    return SimpleNamespace()


def _cand(source, title, url, artist="A", direct_url=""):
    from nomorals.media.sources import SourceCandidate
    return SourceCandidate(source=source, title=title, artist=artist,
                           url=url, direct_url=direct_url or url)


# ── BUG #2: chain continues past non-downloadable ──────────────────────


def _chain_ctx(monkeypatch):
    """All chain sources empty except the ones a test fills in."""
    from nomorals.media import sources as _src
    from nomorals.media import playback as _pb
    for name in ("NetNaijaSource", "NotJustOkSource", "TooXclusiveSource",
                 "BandcampSource", "ArchiveOrgSource", "MusicBrainzSource",
                 "OpenverseSource", "WikimediaSource", "ITunesSource",
                 "DeezerSource", "BoomplaySource"):
        monkeypatch.setattr(getattr(_src, name), "search",
                            lambda self, q, limit=5: [])
    monkeypatch.setattr(
        _pb.PlaybackEngine, "_youtube_search_id",
        classmethod(lambda cls, q, prefix="ytsearch": (_ for _ in ()
                                                       ).throw(
            RuntimeError("youtube search needs yt-dlp"))))
    return SourceResolver(_ctx())


def test_chain_continues_past_boomplay_to_downloadable(monkeypatch):
    from nomorals.media import sources as _src
    r = _chain_ctx(monkeypatch)
    monkeypatch.setattr(
        _src.BoomplaySource, "search",
        lambda self, q, limit=5: [_cand("boomplay", "Boom Song",
                                        "https://boomplay.com/x")])
    monkeypatch.setattr(
        _src.OpenverseSource, "search",
        lambda self, q, limit=5: [_cand("openverse", "Open Song",
                                         "https://example.com/x.mp3")])
    out = r.resolve("boom song")
    assert out.ok
    # Boomplay is metadata-only (not downloadable) — the chain must
    # keep going to Openverse, which CAN deliver audio.
    assert out.kind == "openverse"
    joined = " ".join(out.attempts)
    assert "boomplay-search" in joined
    assert "openverse-search" in joined


def test_chain_returns_nondownloadable_when_nothing_downloadable(
        monkeypatch):
    from nomorals.media import sources as _src
    r = _chain_ctx(monkeypatch)
    monkeypatch.setattr(
        _src.BoomplaySource, "search",
        lambda self, q, limit=5: [_cand("boomplay", "Boom Song",
                                         "https://boomplay.com/x")])
    out = r.resolve("boom song")
    assert out.ok
    assert out.kind == "boomplay"
    assert out.downloadable is False
    # ...but the chain still TRIED everything after Boomplay.
    joined = " ".join(out.attempts)
    for name in ("musicbrainz-search", "openverse-search",
                 "wikimedia-search", "itunes-search", "deezer-search",
                 "spotify-search"):
        assert name in joined, joined


def test_chain_prefers_first_nondownloadable(monkeypatch):
    from nomorals.media import sources as _src
    r = _chain_ctx(monkeypatch)
    monkeypatch.setattr(
        _src.BoomplaySource, "search",
        lambda self, q, limit=5: [_cand("boomplay", "Boom Song",
                                         "https://boomplay.com/x")])
    monkeypatch.setattr(
        _src.MusicBrainzSource, "search",
        lambda self, q, limit=5: [_cand("musicbrainz", "MB Song",
                                         "https://musicbrainz.org/x")])
    out = r.resolve("some song")
    assert out.ok
    assert out.kind == "boomplay"  # first non-downloadable hit wins
    assert out.downloadable is False


# ── fake queue-backed engine for BUG #3 / #4 ───────────────────────────


class _FakeQueueEngine:
    """List-backed queue + scripted downloads."""

    def __init__(self, rows, results):
        self._rows = [dict(r) for r in rows]
        self._results = list(results)
        self.download_calls = []

    def download(self, item):
        self.download_calls.append(item)
        if self._results:
            return self._results.pop(0)
        return {"ok": False, "reason": "no more scripted downloads"}

    def queue(self):
        return [{"index": i, **r} for i, r in enumerate(self._rows)]

    def remove(self, index):
        del self._rows[int(index)]
        return {"removed": True, "queue": len(self._rows)}

    def _enqueue(self, path, kind, title, **kw):
        row = {"path": path, "kind": kind, "title": title,
               "artist": kw.get("artist", ""),
               "duration": kw.get("duration", 0.0)}
        self._rows.append(row)
        return dict(row)


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


# ── BUG #3: fall-through swaps the queued item to the delivered one ────


def test_fallthrough_swaps_queue_to_delivered():
    rt = _handler()
    chat = _chat()
    sc = {"kind": "soundcloud", "path": "https://soundcloud.com/a/b",
          "title": "Song", "artist": "A"}
    yt = {"kind": "youtube", "path": "https://youtube.com/watch?v=xyz12345678",
          "title": "Song", "artist": "A"}
    engine = _FakeQueueEngine(
        [dict(sc)],
        [{"ok": False, "reason": "This video is DRM protected"},
         {"ok": True, "path": "/tmp/yt.mp3", "title": "Song"}])
    out = rt._play_send_with_fallthrough(
        engine, sc, [yt], ["soundcloud-search — ok (1)",
                           "youtube-search — ok (1)"], "queued 1:", chat)
    assert "sent" in out
    rows = engine.queue()
    assert len(rows) == 1
    assert rows[0]["kind"] == "youtube"
    assert rows[0]["path"] == yt["path"]


def test_fallthrough_first_success_leaves_queue():
    rt = _handler()
    chat = _chat()
    sc = {"kind": "soundcloud", "path": "https://soundcloud.com/a/b",
          "title": "Song", "artist": "A"}
    engine = _FakeQueueEngine(
        [dict(sc)],
        [{"ok": True, "path": "/tmp/sc.mp3", "title": "Song"}])
    out = rt._play_send_with_fallthrough(
        engine, sc, [], ["soundcloud-search — ok (1)"], "queued 1:", chat)
    assert "sent" in out
    rows = engine.queue()
    assert len(rows) == 1
    assert rows[0]["kind"] == "soundcloud"


# ── BUG #4: failed single-send removes the broken queued item ─────────


def test_send_failure_removes_broken_item_from_queue():
    rt = _handler()
    chat = _chat()
    item = {"kind": "youtube", "path": "https://youtube.com/watch?v=xyz12345678",
            "title": "Song"}
    engine = _FakeQueueEngine(
        [dict(item)], [{"ok": False, "reason": "video unavailable"}])
    out = rt._play_send_in_chat(engine, item, "queued:", chat)
    assert "couldn't download" in out
    assert engine.queue() == []
    rt.gateway.send_file.assert_not_called()


def test_send_success_keeps_item_in_queue():
    rt = _handler()
    chat = _chat()
    item = {"kind": "youtube", "path": "https://youtube.com/watch?v=xyz12345678",
            "title": "Song"}
    engine = _FakeQueueEngine(
        [dict(item)], [{"ok": True, "path": "/tmp/s.mp3", "title": "Song"}])
    out = rt._play_send_in_chat(engine, item, "queued:", chat)
    assert "sent" in out
    assert len(engine.queue()) == 1


# ── BUG #5: pick-list consumed only on successful pick ──────────────────


class _FakePickEngine:
    downloads = []

    def __init__(self, context):
        pass

    def _enqueue(self, path, kind, title, **kw):
        return {"path": path, "kind": kind, "title": title}

    def download(self, item):
        if _FakePickEngine.downloads:
            return _FakePickEngine.downloads.pop(0)
        return {"ok": False, "reason": "no more scripted downloads"}

    def play(self):
        return {"status": "playing", "current": "x"}

    def queue(self):
        return []

    def remove(self, index):
        return {}


def _wait_play_job(ack: str, timeout: float = 10.0):
    """Collect the background play-download job an ack announced."""
    import re
    from nomorals.core.jobs import shared_registry

    m = re.search(r"\[job (job[0-9a-f]+)\]", ack or "")
    assert m, f"ack carries no job id: {ack!r}"
    job = shared_registry().collect(m.group(1), timeout=timeout)
    assert job is not None and job.done(), \
        f"job never finished: {job.status if job else None}"
    return job


def _followup_texts(rt) -> list:
    """Texts the job body posted to the chat (the synchronous reply
    minus the part the ack already showed)."""
    return [c.args[2] for c in rt.gateway.send.call_args_list
            if len(c.args) > 2]


def test_pick_failed_download_keeps_list():
    PickCache.clear()
    rt = _handler()
    rt._ref_from_key = lambda key: _chat()
    cand = ResolvedAudio(ok=True, path_or_url="https://youtube.com/watch?v=xyz12345678",
                         title="Song", kind="youtube", source_name="YouTube")
    token = PickCache.store([cand], "telegram:12345")
    assert token
    _FakePickEngine.downloads = [{"ok": False, "reason": "DRM protected"}]
    with mock.patch("nomorals.media.playback.PlaybackEngine", _FakePickEngine):
        ack1 = rt._play_pick(f"{token} 1", "telegram:12345")
    assert "downloading in the background" in ack1
    _wait_play_job(ack1)
    assert any("couldn't download" in t for t in _followup_texts(rt))
    # Failed pick must NOT consume the list — the user can pick again.
    assert PickCache.get(token) is not None
    _FakePickEngine.downloads = [{"ok": True, "path": "/tmp/s.mp3",
                                   "title": "Song"}]
    with mock.patch("nomorals.media.playback.PlaybackEngine", _FakePickEngine):
        ack2 = rt._play_pick(f"{token} 1", "telegram:12345")
    _wait_play_job(ack2)
    assert any("sent" in t for t in _followup_texts(rt))
    # Successful pick consumes the list.
    assert PickCache.get(token) is None
    PickCache.clear()


# ── BUG #6: pick-list labels source + preview ──────────────────────────


def test_picklist_shows_source_and_preview():
    cands = [
        SimpleNamespace(title="Full Song", artist="A", duration=200.0,
                        kind="youtube", source_name="YouTube search (yt-dlp)"),
        SimpleNamespace(title="Preview Song", artist="B", duration=30.0,
                        kind="itunes",
                        source_name="ITunesSource search (30s preview)"),
        SimpleNamespace(title="Band Song", artist="C", duration=180.0,
                        kind="bandcamp", source_name="Bandcamp search"),
    ]
    msg = format_picklist(cands, "songs", "a1b2c3d4")
    assert "[YouTube]" in msg
    assert "30s preview" in msg
    assert "[iTunes" in msg
    assert "[Bandcamp]" in msg
    # Button parsing still works: lines start "N. ".
    assert "1. Full Song" in msg
    assert "pick:a1b2c3d4" in msg


# ── BUG #7: candidate sweep excludes non-downloadable ───────────────────


def test_search_candidates_excludes_nondownloadable():
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    mb = ResolvedAudio(ok=True, path_or_url="https://musicbrainz.org/x",
                       title="MB Song", kind="musicbrainz",
                       source_name="MusicBrainz search", downloadable=False)
    yt = ResolvedAudio(ok=True, path_or_url="https://youtube.com/watch?v=xyz12345678",
                       title="YT Song", kind="youtube",
                       source_name="YouTube search (yt-dlp)")
    with mock.patch.object(r, "_musicbrainz_search_many",
                           return_value=[mb]), \
            mock.patch.object(r, "_youtube_search_many",
                               return_value=[yt]), \
            mock.patch.object(r, "_netnaija_search_many", return_value=[]), \
            mock.patch.object(r, "_notjustok_search_many", return_value=[]), \
            mock.patch.object(r, "_tooxclusive_search_many", return_value=[]), \
            mock.patch.object(r, "_soundcloud_search_many", return_value=[]), \
            mock.patch.object(r, "_bandcamp_search_many", return_value=[]), \
            mock.patch.object(r, "_youtube_music_search_many",
                               return_value=[]), \
            mock.patch.object(r, "_archiveorg_search_many", return_value=[]), \
            mock.patch.object(r, "_openverse_search_many", return_value=[]), \
            mock.patch.object(r, "_wikimedia_search_many", return_value=[]), \
            mock.patch.object(r, "_itunes_search_many", return_value=[]), \
            mock.patch.object(r, "_deezer_search_many", return_value=[]):
        cands, attempts = r.search_candidates_detailed("song", limit=8)
    assert [c.title for c in cands] == ["YT Song"]
    # ...but the audit trail still records that MusicBrainz was tried.
    assert any(a.startswith("musicbrainz-search — ok") for a in attempts)


# ── BUG #8: single-send cookies hint on YouTube bot-check ───────────────


def test_single_send_botcheck_appends_short_hint():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": False,
        "reason": ("YouTube download failed: ERROR: [youtube] abc: "
                   "Sign in to confirm you're not a bot.")}
    out = rt._play_send_in_chat(engine, {"title": "X"}, "queued:", chat)
    assert "couldn't download" in out
    assert "/cookies" in out
    assert BOT_COOKIE_HELP not in out
    assert "--cookies-from-browser chrome" not in out


def test_single_send_generic_no_hint():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.return_value = {"ok": False,
                                    "reason": "video unavailable"}
    out = rt._play_send_in_chat(engine, {"title": "X"}, "queued:", chat)
    assert "couldn't download" in out
    assert "/cookies" not in out


# ── BUG #9: forced-YouTube exception paths name the cookies fix ─────────


def test_forced_youtube_chat_botcheck_short_hint():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt._ref_from_key = lambda key: _chat()

    class FakeEngine:
        def __init__(self, context):
            pass

        def _add_youtube(self, target, title=""):
            raise Exception("youtube search failed: ERROR: [youtube] abc: "
                            "Sign in to confirm you're not a bot.")

    with mock.patch("nomorals.media.playback.PlaybackEngine", FakeEngine):
        out = rt._control_play("youtube:lifestyle", "telegram:12345")
    assert "can't play" in out
    assert "/cookies" in out
    assert BOT_COOKIE_HELP not in out
    assert "--cookies-from-browser chrome" not in out


def test_forced_youtube_cli_botcheck_full_help():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt._ref_from_key = lambda key: None

    class FakeEngine:
        def __init__(self, context):
            pass

        def _add_youtube(self, target, title=""):
            raise Exception("youtube search failed: ERROR: [youtube] abc: "
                            "Sign in to confirm you're not a bot.")

    with mock.patch("nomorals.media.playback.PlaybackEngine", FakeEngine):
        out = rt._control_play("youtube:lifestyle", "")
    # CLI users CAN run yt-dlp commands — the full help is actionable.
    assert "can't play" in out
    assert "yt-dlp --cookies-from-browser chrome" in out


# ── BUG #10: "couldn't find" chat path uses the short cookies hint ──────


def test_control_play_not_found_uses_short_cookies_hint():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt._ref_from_key = lambda key: _chat()

    failed = ResolvedAudio(
        ok=False,
        attempts=["workspace-scan — failed",
                  "youtube-search — failed"],
        errors=["youtube-search: youtube search failed: ERROR: "
                "Sign in to confirm you're not a bot."],
        hint=BOT_COOKIE_HELP)

    class FakeResolver:
        def __init__(self, context):
            pass

        def resolve(self, query):
            return failed

        def search_candidates(self, query, limit=8):
            return []

    with mock.patch("nomorals.media.resolver.SourceResolver", FakeResolver), \
            mock.patch("nomorals.media.playback.PlaybackEngine"):
        out = rt._control_play("some obscure song xyz", "telegram:12345")
    assert "couldn't find" in out
    assert "/cookies" in out
    assert BOT_COOKIE_HELP not in out
    assert "--cookies-from-browser chrome" not in out


# ── BUG #11: NL "play X" passes chat_key + honors the music gate ────────


def test_nl_play_passes_chat_key():
    from nomorals.agents import coremind as _cm
    mind = _cm.CoreMind.__new__(_cm.CoreMind)
    mind.runtime = mock.Mock()
    mind.runtime._control_play.return_value = "sent song"
    mind._route_line = lambda intent: ""
    intent = SimpleNamespace(target="lifestyle ya man")
    with mock.patch("nomorals.partner.chat_modes.is_feature_disabled",
                    return_value=False):
        out = mind._dispatch_play(intent, "job", "telegram:12345", None)
    assert "sent song" in out
    args, kwargs = mind.runtime._control_play.call_args
    assert kwargs.get("chat_key") == "telegram:12345"


def test_nl_play_respects_music_off():
    from nomorals.agents import coremind as _cm
    mind = _cm.CoreMind.__new__(_cm.CoreMind)
    mind.runtime = mock.Mock()
    mind.runtime._control_play.return_value = "sent song"
    mind._route_line = lambda intent: ""
    intent = SimpleNamespace(target="lifestyle ya man")
    with mock.patch("nomorals.partner.chat_modes.is_feature_disabled",
                    return_value=True), \
            mock.patch("nomorals.partner.chat_modes.feature_disabled_reply",
                       return_value="Music is turned off in this chat."):
        out = mind._dispatch_play(intent, "job", "telegram:12345", None)
    assert "turned off" in out
    mind.runtime._control_play.assert_not_called()


# ── BUG #12: transport arg validation ───────────────────────────────────


def _control_play_cli(tail):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt._ref_from_key = lambda key: None
    engine = mock.Mock()
    engine.status.return_value = {"queue": 0, "current": "",
                                  "backend": {"name": "none"},
                                  "playing": False, "paused": False,
                                  "position": 0, "volume": 80}
    with mock.patch("nomorals.media.playback.PlaybackEngine",
                    return_value=engine):
        return rt._control_play(tail, ""), engine


def test_play_seek_bad_arg_usage_not_traceback():
    out, _ = _control_play_cli("seek abc")
    assert "play error" not in out
    assert "usage" in out.lower()


def test_play_seek_no_arg_usage():
    out, _ = _control_play_cli("seek")
    assert "play error" not in out
    assert "usage" in out.lower()


def test_play_volume_bad_arg_usage():
    out, _ = _control_play_cli("volume loud")
    assert "play error" not in out
    assert "usage" in out.lower()


def test_play_remove_bad_arg_usage():
    out, _ = _control_play_cli("remove abc")
    assert "play error" not in out
    assert "usage" in out.lower()


def test_play_remove_no_arg_usage():
    out, _ = _control_play_cli("remove")
    assert "play error" not in out
    assert "usage" in out.lower()


def test_play_remove_out_of_range_clean():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt._ref_from_key = lambda key: None
    engine = mock.Mock()
    engine.remove.side_effect = Exception("no queue item 99")
    with mock.patch("nomorals.media.playback.PlaybackEngine",
                    return_value=engine):
        out = rt._control_play("remove 99", "")
    assert "play error" not in out
    assert "no queue item 99" in out


# ── BUG #13: /cookies reports stale files honestly ───────────────────────


def _cookies_file(tmp_path, rows):
    f = tmp_path / "cookies.txt"
    lines = ["# Netscape HTTP Cookie File"]
    for domain, expiry, name in rows:
        lines.append(f"{domain}\tTRUE\t/\tTRUE\t{expiry}\t{name}\tYES")
    f.write_text("\n".join(lines) + "\n")
    return f


def test_cookies_status_counts_valid_youtube_cookies(tmp_path, monkeypatch):
    from nomorals.media.cookies import cookies_status
    f = _cookies_file(tmp_path, [
        (".youtube.com", 9999999999, "CONSENT"),   # valid
        (".youtube.com", 1000000000, "OLD"),       # expired
        (".example.com", 9999999999, "OTHER"),     # wrong domain
    ])
    monkeypatch.setenv("YTDLP_COOKIES", str(f))
    st = cookies_status()
    assert st["found"] == str(f)
    assert st["valid_youtube"] == 1


def test_cookies_status_zero_when_all_expired(tmp_path, monkeypatch):
    from nomorals.media.cookies import cookies_status
    f = _cookies_file(tmp_path, [(".youtube.com", 1000000000, "OLD")])
    monkeypatch.setenv("YTDLP_COOKIES", str(f))
    st = cookies_status()
    assert st["valid_youtube"] == 0


def test_control_cookies_flags_stale_file(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    f = _cookies_file(tmp_path, [(".youtube.com", 1000000000, "OLD")])
    monkeypatch.setenv("YTDLP_COOKIES", str(f))
    out = rt._control_cookies("")
    assert "no valid" in out.lower() or "expired" in out.lower()
    assert "✅ cookies file found" not in out


def test_control_cookies_shows_valid_count(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    f = _cookies_file(tmp_path, [(".youtube.com", 9999999999, "CONSENT")])
    monkeypatch.setenv("YTDLP_COOKIES", str(f))
    out = rt._control_cookies("")
    assert "✅" in out
    assert "1 valid" in out


# ── BUG #14: forced-YouTube CLI double-enqueue ───────────────────────────


def test_forced_youtube_cli_enqueues_once():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt._ref_from_key = lambda key: None

    class FakeEngine:
        add_calls = 0

        def __init__(self, context):
            pass

        def _add_youtube(self, target, title=""):
            FakeEngine.add_calls += 1
            return [{"path": "https://youtube.com/watch?v=xyz12345678",
                     "kind": "youtube", "title": target}]

        def play_youtube(self, target):
            # Real PlaybackEngine.play_youtube re-adds internally.
            self._add_youtube(target)
            return {"status": "playing", "current": target}

        def queue(self):
            return [{"index": 0, "path": "https://youtube.com/watch?v=xyz12345678",
                     "kind": "youtube", "title": "x"}]

        def play(self, index=None):
            return {"status": "playing", "current": "x",
                    "backend": "test"}

    FakeEngine.add_calls = 0
    with mock.patch("nomorals.media.playback.PlaybackEngine", FakeEngine):
        out = rt._control_play("youtube:some song", "")
    assert FakeEngine.add_calls == 1
    assert "playing" in out
