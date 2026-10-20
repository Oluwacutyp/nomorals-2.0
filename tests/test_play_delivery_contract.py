"""The /play delivery contract (owner decision 2026-10-20).

Two live failures in the owner's own chat today:

1. ``/play lifestyle (YA MAN)`` delivered "Ya Man Ain't Me" — a
   namesake that merely shares the parenthetical — presented as the
   delivery, in a file named ``playlist [playlist].mp3``.
2. ``/play lifestyle (YA MAN) by Ayo Maff`` found the exact track on
   SoundCloud, its download was blocked (DRM), and the chain silently
   substituted "Dealer" — a different song entirely.

The owner's contract: report honestly — can't find it, or found it at
X but couldn't download it there — and either send the track that WAS
downloaded (the confident match) or send a list for the owner to
choose from.  NO silent substitution, and NO hard gate either: a
confident match still auto-sends; only non-confident downloads fall
to the pick-list.

Seams are the documented ones (fake engine/gateway/resolver, no
network), mirroring tests/test_play_download_fallthrough.py.
"""

import re
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media.picklist import (
    PickCache, is_confident_match, match_confidence)
from nomorals.media.resolver import ResolvedAudio


# ── fixtures / helpers ────────────────────────────────────────────────


def _handler():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt.gateway.send.return_value = mock.Mock(ok=True)
    rt.gateway.send_file.return_value = mock.Mock(ok=True)
    return rt


def _chat(chat_id="12345"):
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = chat_id
    chat.key = f"telegram:{chat_id}"
    return chat


def _wait_play_job(ack: str, timeout: float = 10.0):
    from nomorals.core.jobs import shared_registry

    m = re.search(r"\[job (job[0-9a-f]+)\]", ack or "")
    assert m, f"ack carries no job id: {ack!r}"
    job = shared_registry().collect(m.group(1), timeout=timeout)
    assert job is not None and job.done(), \
        f"job never finished: {job.status if job else None}"
    return job


def _followup_texts(rt) -> list:
    return [c.args[2] for c in rt.gateway.send.call_args_list
            if len(c.args) > 2]


SC_URL = "https://soundcloud.com/ayorinde-mafoluku/lifestyle-ya-man"
REAL = {"kind": "soundcloud", "path": SC_URL,
        "title": "Lifestyle (YA MAN)", "artist": "Ayo Maff"}
NAMESAKE = {"kind": "soundcloud",
            "path": "https://soundcloud.com/someone/ya-man-aint-me",
            "title": "Ya Man Ain't Me", "artist": "Cover Crew"}
DEALER = {"kind": "itunes", "path": "https://itunes.example/dealer.m4a",
          "title": "Dealer", "artist": "Some Dealer"}
YT_REAL = {"kind": "youtube",
           "path": "https://www.youtube.com/watch?v=abcdefghijk",
           "title": "Lifestyle (YA MAN)", "artist": "Ayo Maff"}

SEARCH_ATTEMPTS = [
    "workspace-scan — failed",
    "netnaija-search — no results",
    "soundcloud-search — ok (7)",
    "youtube-search — ok (3)",
    "itunes-search — ok (8)",
]

MATCH = ("Lifestyle (YA MAN)", "Ayo Maff")


@pytest.fixture(autouse=True)
def _clean_pick_cache():
    PickCache.clear()
    yield
    PickCache.clear()


# ── the matcher: one confidence notion ────────────────────────────────


def test_matcher_namesake_is_not_confident():
    # "Ya Man Ain't Me" shares the parenthetical but is NOT "Lifestyle".
    assert not is_confident_match(
        "Ya Man Ain't Me", "Cover Crew", "Lifestyle (YA MAN)", "Ayo Maff")
    assert not is_confident_match(
        "Ya Man Ain't Me", "", "Lifestyle (YA MAN)", "")
    score = match_confidence(
        "Ya Man Ain't Me", "", "Lifestyle (YA MAN)", "")
    assert score < 0.6


def test_matcher_unrelated_title_is_not_confident():
    assert not is_confident_match(
        "Dealer", "Some Dealer", "Lifestyle (YA MAN)", "Ayo Maff")
    assert match_confidence(
        "Dealer", "Some Dealer", "Lifestyle (YA MAN)", "Ayo Maff") == 0.0


def test_matcher_exact_match_is_confident():
    assert is_confident_match(
        "Lifestyle (YA MAN)", "Ayo Maff", "Lifestyle (YA MAN)", "Ayo Maff")
    assert match_confidence(
        "Lifestyle (YA MAN)", "Ayo Maff",
        "Lifestyle (YA MAN)", "Ayo Maff") == pytest.approx(1.0)


def test_matcher_messy_youtube_title_is_confident():
    # Real-world YouTube titles carry the artist + "(Official Video)".
    assert is_confident_match(
        "Ayo Maff - Lifestyle (YA MAN) (Official Video)", "Ayo Maff",
        "Lifestyle (YA MAN)", "Ayo Maff")


def test_matcher_wrong_artist_is_not_confident():
    # Same title, different artist, and the query named the artist.
    assert not is_confident_match(
        "Lifestyle (YA MAN)", "DJ Kaywise",
        "Lifestyle (YA MAN)", "Ayo Maff")


def test_matcher_partial_title_is_not_confident():
    # Bare "Lifestyle" is a different (or at least unproven) song.
    assert not is_confident_match(
        "Lifestyle", "Ayo Maff", "Lifestyle (YA MAN)", "Ayo Maff")


def test_matcher_no_artist_in_query_title_alone_decides():
    assert is_confident_match(
        "Lifestyle (YA MAN)", "Whoever", "Lifestyle (YA MAN)", "")


# ── (a) exact match found-but-blocked; namesake downloadable ────────


def test_blocked_match_namesake_not_sent_picklist_offered():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.queue.return_value = []
    # Only ONE download may be attempted: the confident SoundCloud
    # match.  The namesake must never even be tried as a delivery.
    engine.download.return_value = {
        "ok": False, "reason": "This video is DRM protected"}
    out = rt._play_send_with_fallthrough(
        engine, dict(REAL), [dict(NAMESAKE)], SEARCH_ATTEMPTS,
        "queued 1:", chat, match=MATCH, query="lifestyle (YA MAN) by Ayo Maff")
    assert engine.download.call_count == 1
    assert engine.download.call_args[0][0]["path"] == SC_URL
    rt.gateway.send_file.assert_not_called()
    # Found-but-blocked is a first-class fact in the report.
    assert "SoundCloud" in out
    assert "DRM" in out
    assert "couldn't download" in out
    # The namesake is offered as a CHOICE, not sent as the delivery.
    assert "Ya Man Ain't Me" in out
    assert re.search(r"pick:[0-9a-f]{8}", out)
    assert "sent “" not in out


def test_blocked_match_without_alternatives_reports_blocks():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.queue.return_value = []
    engine.download.return_value = {
        "ok": False, "reason": "This video is DRM protected"}
    out = rt._play_send_with_fallthrough(
        engine, dict(REAL), [], SEARCH_ATTEMPTS, "queued 1:", chat,
        match=MATCH, query="lifestyle (YA MAN) by Ayo Maff")
    rt.gateway.send_file.assert_not_called()
    assert "couldn't download" in out
    assert "DRM" in out
    assert "soundcloud-search" in out


# ── (b) namesake first, real match later in the same source ──────────


def test_namesake_skipped_real_match_sent():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.queue.return_value = []
    filler1 = {"kind": "soundcloud", "path": "https://soundcloud.com/x/1",
               "title": "Lifestyle Remix (Sped Up)", "artist": "DJ Kaywise"}
    filler2 = {"kind": "soundcloud", "path": "https://soundcloud.com/x/2",
               "title": "Ya Man", "artist": "Someone"}
    engine.download.return_value = {
        "ok": True, "path": "/tmp/real.mp3", "title": "Lifestyle (YA MAN)"}
    items = [dict(NAMESAKE), filler1, filler2, dict(REAL),
             {"kind": "soundcloud", "path": "https://soundcloud.com/x/3",
              "title": "Lifestyle (YA MAN) [reup]", "artist": "Ayo Maff"},
             {"kind": "soundcloud", "path": "https://soundcloud.com/x/4",
              "title": "Unrelated Jam", "artist": "Nobody"},
             {"kind": "soundcloud", "path": "https://soundcloud.com/x/5",
              "title": "Another One", "artist": "Nobody"}]
    out = rt._play_send_with_fallthrough(
        engine, items[0], items[1:], SEARCH_ATTEMPTS, "queued 1:", chat,
        match=MATCH, query="lifestyle (YA MAN) by Ayo Maff")
    attempted = [c[0][0]["path"] for c in engine.download.call_args_list]
    assert NAMESAKE["path"] not in attempted
    assert attempted == [SC_URL]
    rt.gateway.send_file.assert_called_once()
    assert "/tmp/real.mp3" in rt.gateway.send_file.call_args[0]
    assert "sent “Lifestyle (YA MAN)”" in out


# ── (c) blocked on A, downloaded from C — report names A's block ─────


def test_blocked_on_a_sent_from_c_report_names_block():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.queue.return_value = []
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": True, "path": "/tmp/yt-real.mp3",
         "title": "Lifestyle (YA MAN)"},
    ]
    out = rt._play_send_with_fallthrough(
        engine, dict(REAL), [dict(YT_REAL)], SEARCH_ATTEMPTS,
        "queued 1:", chat, match=MATCH,
        query="lifestyle (YA MAN) by Ayo Maff")
    assert engine.download.call_count == 2
    rt.gateway.send_file.assert_called_once()
    assert "/tmp/yt-real.mp3" in rt.gateway.send_file.call_args[0]
    assert "sent “Lifestyle (YA MAN)”" in out
    # The per-source trail carries the found-but-blocked fact.
    assert "soundcloud-search" in out and "DRM" in out
    assert "downloaded ✓" in out


# ── (d) honest terminal reports ───────────────────────────────────────


def test_nothing_found_anywhere_reports_not_found():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    chat = _chat()
    rt._ref_from_key = lambda key: chat
    res = ResolvedAudio(
        ok=False, attempts=["workspace-scan — failed",
                            "soundcloud-search — no results"],
        errors=["soundcloud-search: no results"],
        hint="tried every source — check the title/URL and try again")
    engine = mock.Mock()
    engine.queue.return_value = []
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = res
        out = rt._control_play("some nonexistent song zzq", chat.key)
    assert "couldn't find" in out
    assert "soundcloud-search" in out
    rt.gateway.send_file.assert_not_called()


def test_found_but_blocked_everywhere_names_every_block():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.queue.return_value = []
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": False, "reason": "YouTube download failed: video unavailable"},
    ]
    out = rt._play_send_with_fallthrough(
        engine, dict(REAL), [dict(YT_REAL)], SEARCH_ATTEMPTS,
        "queued 1:", chat, match=MATCH,
        query="lifestyle (YA MAN) by Ayo Maff")
    rt.gateway.send_file.assert_not_called()
    assert "couldn't download" in out
    assert "DRM" in out
    assert "video unavailable" in out
    assert "pick:" not in out  # nothing downloadable to offer


# ── pick-list round-trip: the owner's choice IS delivered ────────────


def test_picklist_choice_delivers_the_chosen_candidate():
    rt = _handler()
    chat = _chat()
    rt._ref_from_key = lambda key: chat
    engine = mock.Mock()
    engine.queue.return_value = []
    engine._enqueue.side_effect = lambda p, k, t, **kw: {
        "kind": k, "path": p, "title": t, "artist": kw.get("artist", "")}
    # First the blocked-match outcome that produces the pick-list...
    engine.download.return_value = {
        "ok": False, "reason": "This video is DRM protected"}
    out = rt._play_send_with_fallthrough(
        engine, dict(REAL), [dict(NAMESAKE)], SEARCH_ATTEMPTS,
        "queued 1:", chat, match=MATCH, query="lifestyle (YA MAN) by Ayo Maff")
    token = re.search(r"pick:([0-9a-f]{8})", out).group(1)
    # ...then the owner picks the namesake ON PURPOSE.
    engine.download.return_value = {
        "ok": True, "path": "/tmp/namesake.mp3", "title": "Ya Man Ain't Me"}
    with mock.patch("nomorals.media.playback.PlaybackEngine",
                    return_value=engine):
        ack = rt._control_play(f"pick {token} 1", chat.key)
    _wait_play_job(ack)
    rt.gateway.send_file.assert_called_once()
    assert "/tmp/namesake.mp3" in rt.gateway.send_file.call_args[0]
    text = "\n".join(_followup_texts(rt))
    assert "Ya Man Ain't Me" in text


# ── (5) delivered identity: caption + filename ────────────────────────


def test_template_filename_replaced_by_real_identity(tmp_path):
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.queue.return_value = []
    landed = tmp_path / "playlist [playlist].mp3"
    landed.write_bytes(b"ID3-fake-audio")
    engine.download.return_value = {
        "ok": True, "path": str(landed), "title": "Lifestyle (YA MAN)"}
    out = rt._play_send_with_fallthrough(
        engine, dict(REAL), [], SEARCH_ATTEMPTS, "queued 1:", chat,
        match=MATCH, query="lifestyle (YA MAN) by Ayo Maff")
    rt.gateway.send_file.assert_called_once()
    sent_path = rt.gateway.send_file.call_args[0][2]
    caption = rt.gateway.send_file.call_args[1]["caption"]
    assert sent_path.endswith("Ayo Maff - Lifestyle (YA MAN).mp3")
    assert "playlist" not in sent_path
    assert caption == "🎵 Lifestyle (YA MAN)"
    assert "sent “Lifestyle (YA MAN)”" in out


def test_good_filename_sent_as_is(tmp_path):
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.queue.return_value = []
    landed = tmp_path / "Lifestyle (YA MAN) [2361175013].mp3"
    landed.write_bytes(b"ID3-fake-audio")
    engine.download.return_value = {
        "ok": True, "path": str(landed), "title": "Lifestyle (YA MAN)"}
    rt._play_send_with_fallthrough(
        engine, dict(REAL), [], SEARCH_ATTEMPTS, "queued 1:", chat,
        match=MATCH, query="lifestyle (YA MAN) by Ayo Maff")
    sent_path = rt.gateway.send_file.call_args[0][2]
    assert sent_path == str(landed)


def test_single_send_path_also_carries_identity(tmp_path):
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.queue.return_value = []
    landed = tmp_path / "playlist [playlist].mp3"
    landed.write_bytes(b"ID3-fake-audio")
    engine.download.return_value = {
        "ok": True, "path": str(landed), "title": "Lifestyle (YA MAN)"}
    text, sent = rt._play_send_in_chat_result(
        engine, dict(REAL), "queued 1:", chat)
    assert sent is True
    sent_path = rt.gateway.send_file.call_args[0][2]
    assert sent_path.endswith("Ayo Maff - Lifestyle (YA MAN).mp3")


# ── end-to-end: today's two incidents through _control_play ──────────


def _e2e_handler(resolution, candidates, attempts):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt.gateway.send.return_value = mock.Mock(ok=True)
    rt.gateway.send_file.return_value = mock.Mock(ok=True)
    chat = _chat()
    rt._ref_from_key = lambda key: chat
    engine = mock.Mock()
    engine._enqueue.side_effect = lambda p, k, t, **kw: {
        "kind": k, "path": p, "title": t, "artist": kw.get("artist", "")}
    engine.queue.return_value = [{"title": "x"}]
    return rt, chat, engine


def test_e2e_blocked_exact_match_no_substitution():
    """Incident #2: exact on SoundCloud (blocked) + 'Dealer' on iTunes."""
    sc_res = ResolvedAudio(
        ok=True, path_or_url=SC_URL, title="Lifestyle (YA MAN)",
        artist="Ayo Maff", kind="soundcloud",
        source_name="SoundCloud search",
        attempts=["workspace-scan — failed",
                  "soundcloud-search — ok (SoundCloud search)"])
    sc_cand = ResolvedAudio(
        ok=True, path_or_url=SC_URL, title="Lifestyle (YA MAN)",
        artist="Ayo Maff", kind="soundcloud",
        source_name="SoundCloud search")
    dealer_cand = ResolvedAudio(
        ok=True, path_or_url=DEALER["path"], title="Dealer",
        artist="Some Dealer", kind="itunes",
        source_name="iTunes search (30s preview)")
    attempts = ["netnaija-search — no results",
                "soundcloud-search — ok (1)",
                "itunes-search — ok (8)"]
    rt, chat, engine = _e2e_handler(sc_res, None, None)
    engine.download.return_value = {
        "ok": False, "reason": "This video is DRM protected"}
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = sc_res
        SR.return_value.search_candidates_detailed.return_value = (
            [sc_cand, dealer_cand], attempts)
        ack = rt._control_play("lifestyle (YA MAN) by Ayo Maff", chat.key)
    _wait_play_job(ack)
    # 'Dealer' was never attempted and never sent.
    assert engine.download.call_count == 1
    rt.gateway.send_file.assert_not_called()
    text = "\n".join(_followup_texts(rt))
    assert "DRM" in text
    assert "Dealer" in text  # offered as a pick, not delivered
    assert re.search(r"pick:[0-9a-f]{8}", text)


def test_e2e_namesake_chain_pick_real_match_sent():
    """Incident #1: the chain's first pick is the namesake; the real
    match sits in the candidate sweep and downloads — IT is sent."""
    sc_res = ResolvedAudio(
        ok=True, path_or_url=NAMESAKE["path"], title="Ya Man Ain't Me",
        artist="Cover Crew", kind="soundcloud",
        source_name="SoundCloud search",
        attempts=["workspace-scan — failed",
                  "soundcloud-search — ok (SoundCloud search)"])
    ns_cand = ResolvedAudio(
        ok=True, path_or_url=NAMESAKE["path"], title="Ya Man Ain't Me",
        artist="Cover Crew", kind="soundcloud",
        source_name="SoundCloud search")
    yt_cand = ResolvedAudio(
        ok=True, path_or_url=YT_REAL["path"], title="Lifestyle (YA MAN)",
        artist="Ayo Maff", kind="youtube",
        source_name="YouTube search (yt-dlp)")
    attempts = ["soundcloud-search — ok (7)", "youtube-search — ok (3)"]
    rt, chat, engine = _e2e_handler(sc_res, None, None)
    engine.download.return_value = {
        "ok": True, "path": "/tmp/yt-real.mp3", "title": "Lifestyle (YA MAN)"}
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = sc_res
        SR.return_value.search_candidates_detailed.return_value = (
            [ns_cand, yt_cand], attempts)
        ack = rt._control_play("lifestyle (YA MAN)", chat.key)
    _wait_play_job(ack)
    attempted = [c[0][0]["path"] for c in engine.download.call_args_list]
    assert attempted == [YT_REAL["path"]]
    rt.gateway.send_file.assert_called_once()
    assert "/tmp/yt-real.mp3" in rt.gateway.send_file.call_args[0]
    text = "\n".join(_followup_texts(rt))
    assert "sent “Lifestyle (YA MAN)”" in text
