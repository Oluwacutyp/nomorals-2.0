"""Preview law (owner screenshot, 2026-11-05): /play fear by nf.

The bug: the resolver chain WON on iTunes, whose only audio asset is
a 30-second AudioPreview clip; the chat path then queued a download
job for the preview, failed to fetch it like a song, and offered a
pick-list of five more "[iTunes · 30s preview]" entries as if they
were tracks.  Owner: "still downloading previews instead of full
song".

The law pinned here:
- iTunes/Deezer candidates are preview-class from the source's own
  fields (iTunes ``previewUrl`` / Deezer ``preview`` — the ONLY audio
  fields those APIs return): never download candidates, never
  pick-list entries.  They identify a track; they never supply audio.
- The full-track ladder wins when any full source has the track.
- A previews-only world gets the honest absence reply and ZERO
  download jobs — queueing a job for a preview is the defect.
- A download that arrives preview-length for a full-length track is
  a FAILED download (duration verified on arrival), never sent.
- Reply copy carries no raw asset URLs and no doubled failure
  wording.  (The tried-list and the [job <id>] ack reference are a
  separate, deliberate owner design and are NOT touched here.)
"""

import re
import time
from types import SimpleNamespace
from unittest import mock

from nomorals.media.picklist import PickCache
from nomorals.media.resolver import ResolvedAudio, SourceResolver

PREVIEW_URL = (
    "https://audio-ssl.itunes.apple.com/itunes-assets-autolink/"
    "AudioPreview211/v4/ab/cd/fear.m4a")
ITUNES_PAGE = "https://music.apple.com/us/song/fear/1440857781"
DEEZER_PREVIEW_URL = "https://cdns-preview-9.dzcdn.net/stream/fear.mp3"
YT_URL = "https://www.youtube.com/watch?v=abcdefghijk"

ITUNES_PAYLOAD = (
    '{"results": [{"trackName": "FEAR", "artistName": "NF",'
    f' "previewUrl": "{PREVIEW_URL}",'
    f' "trackViewUrl": "{ITUNES_PAGE}",'
    ' "trackTimeMillis": 213000}]}')
DEEZER_PAYLOAD = (
    '{"data": [{"title": "FEAR",'
    f' "preview": "{DEEZER_PREVIEW_URL}",'
    ' "link": "https://www.deezer.com/track/123",'
    ' "artist": {"name": "NF"}, "duration": 213}]}')


def _resolver():
    return SourceResolver(SimpleNamespace(), soundcloud=None,
                          spotify=None)


# ── source level: the asset IS a preview, proven by the fields ───────


def test_itunes_candidate_is_preview_class_not_downloadable():
    from nomorals.media.sources import ITunesSource, candidate_asset_class

    with mock.patch.object(ITunesSource, "_get",
                           return_value=ITUNES_PAYLOAD):
        rows = ITunesSource().search("nf fear")
    assert len(rows) == 1
    c = rows[0]
    # The only audio field iTunes returns is previewUrl (30s); the
    # full trackTimeMillis is identification metadata, not audio.
    assert c.downloadable is False
    assert c.extra["preview_only"] is True
    assert candidate_asset_class(c) == "preview"
    assert c.duration == 213.0


def test_deezer_candidate_is_preview_class_not_downloadable():
    from nomorals.media.sources import DeezerSource, candidate_asset_class

    with mock.patch.object(DeezerSource, "_get",
                           return_value=DEEZER_PAYLOAD):
        rows = DeezerSource().search("nf fear")
    assert len(rows) == 1
    c = rows[0]
    assert c.downloadable is False
    assert c.extra["preview_only"] is True
    assert candidate_asset_class(c) == "preview"
    assert c.duration == 213.0


def test_candidate_asset_class_metadata_and_full():
    from nomorals.media.sources import SourceCandidate, candidate_asset_class

    meta = SourceCandidate(
        title="FEAR", artist="NF", url="https://musicbrainz.org/x",
        source="musicbrainz", downloadable=False)
    assert candidate_asset_class(meta) == "metadata"
    full = SourceCandidate(
        title="FEAR", artist="NF", url=YT_URL, source="youtube")
    assert candidate_asset_class(full) == "full"


# ── resolver level: chain + sweep never surface a preview as audio ──


def test_preview_strategy_is_found_but_blocked():
    from nomorals.media.sources import ITunesSource

    r = _resolver()
    with mock.patch.object(ITunesSource, "_get",
                           return_value=ITUNES_PAYLOAD):
        res = r._s_preview_search("nf fear", "ITunesSource", "itunes")
    assert res.ok is True
    assert res.downloadable is False
    assert res.preview_only is True
    # Identification points at the store PAGE — never the preview
    # asset — so nothing downstream can fetch the clip by accident.
    assert res.path_or_url == ITUNES_PAGE
    assert "AudioPreview" not in res.path_or_url


def test_chain_never_wins_on_a_preview():
    r = _resolver()
    preview = ResolvedAudio(
        ok=True, path_or_url=ITUNES_PAGE, title="FEAR", artist="NF",
        duration=213.0, kind="itunes", downloadable=False,
        preview_only=True)
    full = ResolvedAudio(
        ok=True, path_or_url=YT_URL, title="FEAR", artist="NF",
        duration=213.0, kind="youtube")

    def fail():
        raise RuntimeError("no results")

    base = ResolvedAudio(ok=False)
    out = r._run_chain(base, [("a-search", fail),
                              ("itunes-search", lambda: preview),
                              ("youtube-search", lambda: full)])
    assert out is full or (out.ok and out.kind == "youtube")
    # Preview alone: the chain result is the blocked preview shape,
    # so the caller can report honestly instead of downloading it.
    base2 = ResolvedAudio(ok=False)
    out2 = r._run_chain(base2, [("a-search", fail),
                                ("itunes-search", lambda: preview)])
    assert out2.ok is True
    assert out2.downloadable is False
    assert out2.preview_only is True


def test_sweep_excludes_previews_keeps_full_and_attempts():
    r = _resolver()
    itunes_hit = ResolvedAudio(
        ok=True, path_or_url=ITUNES_PAGE, title="FEAR", artist="NF",
        duration=213.0, kind="itunes", source_name="ITunesSource search",
        downloadable=False, preview_only=True)
    yt_hit = ResolvedAudio(
        ok=True, path_or_url=YT_URL, title="FEAR", artist="NF",
        duration=213.0, kind="youtube",
        source_name="YouTube search (yt-dlp)")
    with mock.patch.object(r, "_itunes_search_many",
                           return_value=[itunes_hit]), \
            mock.patch.object(r, "_deezer_search_many", return_value=[]), \
            mock.patch.object(r, "_youtube_search_many",
                              return_value=[yt_hit]), \
            mock.patch.object(r, "_netnaija_search_many", return_value=[]), \
            mock.patch.object(r, "_notjustok_search_many", return_value=[]), \
            mock.patch.object(r, "_tooxclusive_search_many",
                              return_value=[]), \
            mock.patch.object(r, "_soundcloud_search_many",
                              return_value=[]), \
            mock.patch.object(r, "_bandcamp_search_many", return_value=[]), \
            mock.patch.object(r, "_youtube_music_search_many",
                              return_value=[]), \
            mock.patch.object(r, "_archiveorg_search_many",
                              return_value=[]), \
            mock.patch.object(r, "_musicbrainz_search_many",
                              return_value=[]), \
            mock.patch.object(r, "_openverse_search_many", return_value=[]), \
            mock.patch.object(r, "_wikimedia_search_many",
                              return_value=[]):
        cands, attempts = r.search_candidates_detailed("nf fear", limit=8)
    assert [c.kind for c in cands] == ["youtube"]
    joined = " | ".join(attempts)
    assert "itunes-search" in joined  # the source was still consulted


# ── runtime harness ───────────────────────────────────────────────────


def _handler():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    return rt


def _chat(chat_id="12345"):
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = chat_id
    return chat


def _wait_play_job(chat_key, timeout=10.0):
    from nomorals.core.jobs import shared_registry

    deadline = time.time() + timeout
    while time.time() < deadline:
        for snap in shared_registry().list():
            labels = snap.get("labels") or {}
            if labels.get("owner") == "play" \
                    and labels.get("chat") == chat_key:
                job = shared_registry().collect(
                    snap["id"], timeout=max(0.1, deadline - time.time()))
                if job is not None and job.done():
                    return job
        time.sleep(0.05)
    raise AssertionError("play job never appeared/finished")


def _followup_texts(rt):
    return [c.args[2] for c in rt.gateway.send.call_args_list
            if len(c.args) > 2]


def _preview_resolution():
    return ResolvedAudio(
        ok=True, path_or_url=ITUNES_PAGE, title="FEAR", artist="NF",
        duration=213.0, kind="itunes",
        source_name="ITunesSource search (30s preview)",
        downloadable=False, preview_only=True,
        attempts=["youtube-search — no results",
                  "itunes-search — ok (ITunesSource search (30s preview))"])


def _engine(queue_rows):
    engine = mock.Mock()
    engine._enqueue.side_effect = lambda p, k, t, **kw: {
        "kind": k, "path": p, "title": t,
        "artist": kw.get("artist", ""), "duration": kw.get("duration", 0.0)}
    engine.queue.return_value = queue_rows
    return engine


# ── previews-only world: honest absence, zero download jobs ──────────


def test_previews_only_world_no_job_no_download():
    PickCache.clear()
    rt = _handler()
    chat = _chat("99001")
    rt._ref_from_key = lambda key: chat
    engine = _engine([{"kind": "itunes", "path": ITUNES_PAGE,
                       "title": "FEAR"}])
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = _preview_resolution()
        SR.return_value.search_candidates_detailed.return_value = ([], [
            "youtube-search — no results", "itunes-search — ok (1)"])
        reply = rt._control_play("fear by nf", "telegram:99001")
    # The honest absence line, synchronously — no background job was
    # queued for a preview-grade candidate.
    assert "only as a 30-second preview" in reply
    assert "No source has the full track" in reply
    assert "downloading in the background" not in reply
    engine.download.assert_not_called()
    rt.gateway.send_file.assert_not_called()
    assert PREVIEW_URL not in reply
    PickCache.clear()


# ── incident fixture: preview + full track → the full track lands ────


def test_incident_preview_plus_full_delivers_full_only():
    PickCache.clear()
    rt = _handler()
    chat = _chat("99002")
    rt._ref_from_key = lambda key: chat
    yt_cand = ResolvedAudio(
        ok=True, path_or_url=YT_URL, title="FEAR", artist="NF",
        duration=213.0, kind="youtube",
        source_name="YouTube search (yt-dlp)")
    itunes_cand = ResolvedAudio(
        ok=True, path_or_url=ITUNES_PAGE, title="FEAR", artist="NF",
        duration=213.0, kind="itunes",
        source_name="ITunesSource search (30s preview)",
        downloadable=False, preview_only=True)
    engine = _engine([{"kind": "itunes", "path": ITUNES_PAGE,
                       "title": "FEAR"}])
    engine.download.return_value = {
        "ok": True, "path": "/tmp/fear-full.mp3", "title": "FEAR"}
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = _preview_resolution()
        SR.return_value.search_candidates_detailed.return_value = (
            [itunes_cand, yt_cand],
            ["youtube-search — ok (1)", "itunes-search — ok (1)"])
        ack = rt._control_play("fear by nf", "telegram:99002")
    assert "downloading in the background" in ack
    _wait_play_job("telegram:99002")
    # Exactly one download attempt — the full-track candidate.  The
    # preview was never attempted.
    assert engine.download.call_count == 1
    attempted = engine.download.call_args[0][0]
    assert attempted["kind"] == "youtube"
    assert attempted["path"] == YT_URL
    rt.gateway.send_file.assert_called_once()
    assert "/tmp/fear-full.mp3" in rt.gateway.send_file.call_args[0]
    texts = ack + "\n" + "\n".join(_followup_texts(rt))
    assert PREVIEW_URL not in texts
    assert "AudioPreview" not in texts
    PickCache.clear()


# ── fallthrough: previews never attempted, never pick-listed ─────────


def test_fallthrough_preview_match_plus_namesake_picklist_is_clean():
    PickCache.clear()
    rt = _handler()
    chat = _chat("99003")
    chat.key = "telegram:99003"
    preview = {"kind": "itunes", "path": ITUNES_PAGE, "title": "FEAR",
               "artist": "NF", "duration": 213.0, "downloadable": False,
               "preview_only": True}
    other = {"kind": "youtube", "path": YT_URL,
             "title": "Midnight Drive", "artist": "Someone Else",
             "duration": 180.0}
    engine = mock.Mock()
    engine.queue.return_value = []
    out = rt._play_send_with_fallthrough(
        engine, preview, [other], ["itunes-search — ok (1)",
                                   "youtube-search — ok (1)"],
        "queued 1:", chat, match=("FEAR", "NF"), query="fear by nf")
    # Nothing was download-attempted: the match is a preview, the
    # other result doesn't match the request.
    engine.download.assert_not_called()
    rt.gateway.send_file.assert_not_called()
    # The user hears the preview truth…
    assert "30-second preview" in out
    # …and the pick-list holds ONLY the genuine full-track candidate.
    assert "Midnight Drive" in out
    assert "30s preview" not in out
    assert PREVIEW_URL not in out
    assert not re.search(r"\[iTunes[^\]]*\]\s*FEAR", out)
    PickCache.clear()


def test_fallthrough_all_previews_honest_absence():
    rt = _handler()
    chat = _chat("99004")
    preview = {"kind": "deezer", "path": "https://www.deezer.com/track/1",
               "title": "FEAR", "artist": "NF", "duration": 213.0,
               "downloadable": False, "preview_only": True}
    engine = mock.Mock()
    engine.queue.return_value = []
    out = rt._play_send_with_fallthrough(
        engine, preview, [], ["deezer-search — ok (1)"], "queued 1:",
        chat, match=("FEAR", "NF"), query="fear by nf")
    engine.download.assert_not_called()
    rt.gateway.send_file.assert_not_called()
    assert "only as a 30-second preview" in out
    assert "No source has the full track" in out
    assert DEEZER_PREVIEW_URL not in out


# ── duration verification: a clip is a failed download ───────────────


def test_verify_full_track_flags_preview_length_file():
    rt = _handler()
    item = {"title": "FEAR", "duration": 213.0}
    with mock.patch("nomorals.media.library.read_metadata",
                    return_value={"duration": 30.0}):
        reason = rt._play_verify_full_track("/tmp/x.mp3", item)
    assert "only 30s long" in reason
    assert "not the full 213s track" in reason
    with mock.patch("nomorals.media.library.read_metadata",
                    return_value={"duration": 205.0}):
        assert rt._play_verify_full_track("/tmp/x.mp3", item) == ""
    # Nothing to verify against → no fabricated failure.
    assert rt._play_verify_full_track(
        "/tmp/x.mp3", {"title": "FEAR", "duration": 0.0}) == ""


def test_fallthrough_clip_download_falls_through_to_real_track():
    rt = _handler()
    chat = _chat("99005")
    clip = {"kind": "youtube", "path": YT_URL, "title": "FEAR",
            "artist": "NF", "duration": 213.0}
    real = {"kind": "archiveorg", "path": "https://archive.org/fear.mp3",
            "title": "FEAR", "artist": "NF", "duration": 213.0}
    engine = mock.Mock()
    engine.queue.return_value = []
    engine.download.side_effect = [
        {"ok": True, "path": "/tmp/clip.mp3", "title": "FEAR"},
        {"ok": True, "path": "/tmp/full.mp3", "title": "FEAR"},
    ]

    def _meta(path):
        return {"duration": 30.0 if "clip" in path else 210.0}

    with mock.patch("nomorals.media.library.read_metadata",
                    side_effect=_meta):
        out = rt._play_send_with_fallthrough(
            engine, clip, [real], ["youtube-search — ok (1)"],
            "queued 1:", chat)
    assert engine.download.call_count == 2
    rt.gateway.send_file.assert_called_once()
    assert "/tmp/full.mp3" in rt.gateway.send_file.call_args[0]
    assert "sent “FEAR”" in out


def test_fallthrough_clip_only_candidate_is_failure_not_sent():
    rt = _handler()
    chat = _chat("99006")
    clip = {"kind": "youtube", "path": YT_URL, "title": "FEAR",
            "artist": "NF", "duration": 213.0}
    engine = mock.Mock()
    engine.queue.return_value = []
    engine.download.return_value = {
        "ok": True, "path": "/tmp/clip.mp3", "title": "FEAR"}
    with mock.patch("nomorals.media.library.read_metadata",
                    return_value={"duration": 30.0}):
        out = rt._play_send_with_fallthrough(
            engine, clip, [], ["youtube-search — ok (1)"], "queued 1:",
            chat)
    rt.gateway.send_file.assert_not_called()
    assert "couldn't download" in out
    assert "clip, not the track" in out


# ── reply copy cleanup: URLs out, doubled wording out ────────────────


def test_clean_reason_strips_urls_and_doubling():
    rt = _handler()
    messy = (f"download failed: download failed: fetch "
             f"{PREVIEW_URL} broke")
    clean = rt._play_clean_reason(messy)
    assert "audio-ssl" not in clean
    assert "download failed: download failed:" not in clean
    assert clean.startswith("download failed:")


def test_blocked_line_has_no_raw_url():
    rt = _handler()
    item = {"kind": "soundcloud", "title": "FEAR", "artist": "NF"}
    line = rt._play_blocked_line(
        item, f"download blocked: see {PREVIEW_URL} for details")
    assert "audio-ssl" not in line
    assert "SoundCloud" in line
    assert "couldn't download it" in line


# ── yt-dlp self-provisioning seam ─────────────────────────────────────


def test_ensure_ytdlp_never_installs_under_tests(monkeypatch):
    import nomorals.media.resolver as resolver

    monkeypatch.setattr(resolver, "ytdlp_available", lambda: False)
    monkeypatch.setattr(resolver, "_YTDLP_PROVISION_ATTEMPTED", False)
    run = mock.Mock()
    monkeypatch.setattr("subprocess.run", run)
    # pytest is in sys.modules here: provisioning must not mutate the
    # test environment, and must report honestly.
    assert resolver.ensure_ytdlp() is False
    run.assert_not_called()


def test_ensure_ytdlp_provisions_once_and_rechecks(monkeypatch):
    import sys

    import nomorals.media.resolver as resolver

    state = {"ok": False}
    monkeypatch.setattr(resolver, "ytdlp_available", lambda: state["ok"])
    monkeypatch.setattr(resolver, "_YTDLP_PROVISION_ATTEMPTED", False)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)

    def _fake_run(*a, **kw):
        state["ok"] = True
        return SimpleNamespace(returncode=0)

    run = mock.Mock(side_effect=_fake_run)
    monkeypatch.setattr("subprocess.run", run)
    assert resolver.ensure_ytdlp() is True
    assert run.call_count == 1
    # One bounded attempt per process: a second failure doesn't loop.
    state["ok"] = False
    monkeypatch.setattr(resolver, "_YTDLP_PROVISION_ATTEMPTED", True)
    assert resolver.ensure_ytdlp() is False
    assert run.call_count == 1
