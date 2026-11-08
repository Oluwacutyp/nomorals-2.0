"""Doc 7 wiring (adopted 2026-11-08): the registry play branches run
the REAL search/download pipeline.

Before this build, ``/play registry <q>`` and ``/music registry-play
<q>`` called ``play_via_registry`` with NO ``search_fn`` / ``download_fn``
— the branch answered ``no_search`` (no --source) or ``plan_only``
(--source on a fetch class): metadata and plans presented as results.
These tests drive the ACTUAL command handlers with the codebase's
established fake seams (``nomorals.media.resolver.SourceResolver`` and
``nomorals.media.playback.PlaybackEngine`` patched, a mock gateway)
and pin:

* registry play searches via the real resolver seam and DELIVERS the
  fetched file to the chat as audio;
* a failed first download falls through to the next candidate, and a
  preview-length clip is rejected by the full-track verdict and the
  fallback delivers instead;
* plan/checkout outcomes survive ONLY where the source class makes
  them genuine (purchase → checkout, metadata/preview → no file
  exists) — never as a wiring gap;
* the full-track verdict is three-valued — verified / rejected /
  UNVERIFIED: the registry journey's reporting surfaces label the
  unverified state explicitly, a preview-length arrival is refused
  rather than delivered, and no delivery line claims a verification
  the state doesn't support.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest import mock

from nomorals.media.resolver import ResolvedAudio


# ── fakes at the established seams ────────────────────────────────────


def _audio_file(tmp_path, name="song.mp3", marker=b""):
    p = tmp_path / name
    p.write_bytes(b"\xff\xfb" + marker + b"\x00" * 4096)
    return str(p)


def _cand(kind, url, title="Song", artist="Artist", duration=0.0, **kw):
    return ResolvedAudio(ok=True, path_or_url=url, title=title,
                         artist=artist, duration=duration, kind=kind,
                         **kw)


class _FakeEngine:
    """PlaybackEngine seam: download(item) from a per-URL table."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def download(self, item):
        self.calls.append(dict(item))
        return self._results[item["path"]]


@contextlib.contextmanager
def _patch_pipeline(cands, engine):
    resolver = mock.Mock()
    resolver.search_candidates.return_value = list(cands)
    with mock.patch("nomorals.media.resolver.SourceResolver",
                    lambda ctx: resolver), \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       lambda ctx: engine):
        yield


def _runtime(chat=True):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt.gateway.send_file.return_value = SimpleNamespace(ok=True)
    if chat:
        ref = SimpleNamespace(platform="telegram", chat_id="42")
        rt._ref_from_key = lambda key: ref
    else:
        rt._ref_from_key = lambda key: None
    return rt


# ── /play registry: search + deliver ──────────────────────────────────


def test_play_registry_searches_and_delivers_in_chat(tmp_path):
    path = _audio_file(tmp_path)
    engine = _FakeEngine({"https://youtu.be/x": {
        "ok": True, "path": path, "title": "Song"}})
    rt = _runtime(chat=True)
    with _patch_pipeline([_cand("youtube", "https://youtu.be/x")], engine):
        text = rt._control_play("registry Song", "telegram:42")
    # the fetched file went to the chat as audio — not a text plan
    assert rt.gateway.send_file.call_count == 1
    sent_path = rt.gateway.send_file.call_args[0][2]
    assert sent_path.endswith(".mp3")
    assert "Song" in text
    assert "no_search" not in text and "plan only" not in text
    # candidate duration unknown → the delivery is labelled unverified,
    # never passed off as full-track proof
    assert "unverified" in text


def test_play_registry_cli_names_the_file_not_no_search(tmp_path):
    path = _audio_file(tmp_path)
    engine = _FakeEngine({"https://youtu.be/x": {
        "ok": True, "path": path, "title": "Song"}})
    rt = _runtime(chat=False)
    with _patch_pipeline([_cand("youtube", "https://youtu.be/x")], engine):
        text = rt._control_play("registry Song", "")
    assert "no_search" not in text
    assert "no metadata search wired" not in text
    assert path in text


def test_play_registry_fallback_after_download_failure(tmp_path):
    path = _audio_file(tmp_path)
    engine = _FakeEngine({
        "https://soundcloud.example/t": {"ok": False, "reason": "boom"},
        "https://youtu.be/x": {"ok": True, "path": path, "title": "Song"},
    })
    rt = _runtime(chat=True)
    cands = [_cand("soundcloud", "https://soundcloud.example/t"),
             _cand("youtube", "https://youtu.be/x")]
    with _patch_pipeline(cands, engine):
        text = rt._control_play("registry Song", "telegram:42")
    # first source failed to download → the fallback route delivered
    assert [c["path"] for c in engine.calls] == [
        "https://soundcloud.example/t", "https://youtu.be/x"]
    assert rt.gateway.send_file.call_count == 1
    assert "Song" in text


def test_play_registry_fallback_after_preview_clip(tmp_path):
    clip = _audio_file(tmp_path, "clip.mp3", marker=b"CLIP")
    full = _audio_file(tmp_path, "full.mp3", marker=b"FULL")
    engine = _FakeEngine({
        "https://soundcloud.example/t": {
            "ok": True, "path": clip, "title": "Song"},
        "https://youtu.be/x": {"ok": True, "path": full, "title": "Song"},
    })
    rt = _runtime(chat=True)
    cands = [_cand("soundcloud", "https://soundcloud.example/t",
                   duration=300.0),
             _cand("youtube", "https://youtu.be/x")]

    def fake_meta(p):
        return {"duration": 25.0} if p == clip else {"duration": 0.0}

    with _patch_pipeline(cands, engine), \
            mock.patch("nomorals.media.library.read_metadata", fake_meta):
        rt._control_play("registry Song", "telegram:42")
    # the 25s clip for a 300s track was rejected; the fallback
    # delivered (delivery copies are renamed by title/artist, so the
    # proof is the sent file's CONTENT, not its name)
    assert rt.gateway.send_file.call_count == 1
    sent_path = rt.gateway.send_file.call_args[0][2]
    with open(sent_path, "rb") as fh:
        sent_bytes = fh.read()
    assert b"FULL" in sent_bytes and b"CLIP" not in sent_bytes


# ── plan outcomes survive only where the class makes them genuine ────


def test_play_registry_purchase_source_stays_checkout(tmp_path):
    engine = _FakeEngine({})
    rt = _runtime(chat=True)
    with _patch_pipeline([], engine):
        text = rt._control_play("registry Song --source bandcamp",
                                "telegram:42")
    assert "checkout" in text
    assert engine.calls == []  # a purchase is never auto-downloaded
    assert rt.gateway.send_file.call_count == 0


def test_play_registry_metadata_source_stays_plan(tmp_path):
    engine = _FakeEngine({})
    rt = _runtime(chat=True)
    with _patch_pipeline([], engine):
        text = rt._control_play("registry Song --source musicbrainz",
                                "telegram:42")
    # metadata sources serve no file — the plan is the honest outcome,
    # labelled with the reason, and nothing is delivered
    assert "do not serve files" in text
    assert engine.calls == []
    assert rt.gateway.send_file.call_count == 0


# ── /music registry-play: same wiring, text surface ───────────────────


def test_music_registry_play_delivers_via_callback(tmp_path):
    from nomorals.music_sources.chat import control_music_sources

    path = _audio_file(tmp_path)
    engine = _FakeEngine({"https://youtu.be/x": {
        "ok": True, "path": path, "title": "Song"}})
    delivered = []

    def deliver_fn(p, caption):
        delivered.append((p, caption))
        return True, ""

    with _patch_pipeline([_cand("youtube", "https://youtu.be/x")], engine):
        text = control_music_sources(
            "registry-play Song", context=SimpleNamespace(),
            deliver_fn=deliver_fn)
    assert [p for p, _c in delivered] == [path]
    assert "plan only" not in text and "no_search" not in text


def test_music_registry_play_text_surface_is_honest(tmp_path):
    from nomorals.music_sources.chat import control_music_sources

    path = _audio_file(tmp_path)
    engine = _FakeEngine({"https://youtu.be/x": {
        "ok": True, "path": path, "title": "Song"}})
    with _patch_pipeline([_cand("youtube", "https://youtu.be/x")], engine):
        text = control_music_sources(
            "registry-play Song", context=SimpleNamespace())
    # the file WAS fetched; this surface returns text, so it names the
    # saved file instead of claiming a delivery that never happened
    assert path in text
    assert "saved on the server" in text
    assert "plan only" not in text


def test_control_music_registry_play_passes_context_and_delivers(tmp_path):
    path = _audio_file(tmp_path)
    engine = _FakeEngine({"https://youtu.be/x": {
        "ok": True, "path": path, "title": "Song"}})
    rt = _runtime(chat=True)
    with _patch_pipeline([_cand("youtube", "https://youtu.be/x")], engine):
        text = rt._control_music("registry-play Song", "telegram:42")
    assert rt.gateway.send_file.call_count == 1
    assert "Song" in text


# ── AA-57 outcome recording survives the wiring ───────────────────────


def _real_registry(tmp_path):
    from nomorals.media.source_registry import _seed
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return _seed(store=KVStore(db))


def test_wired_play_records_outcome_in_registry(tmp_path):
    from nomorals.music_sources import play as play_mod
    from nomorals.music_sources import registry_bridge as bridge
    from nomorals.music_sources.live import live_collaborators

    path = _audio_file(tmp_path)
    engine = _FakeEngine({"https://soundcloud.example/t": {
        "ok": True, "path": path, "title": "Song"}})
    reg = _real_registry(tmp_path)
    search_fn, download_fn = live_collaborators(SimpleNamespace())
    with _patch_pipeline(
            [_cand("soundcloud", "https://soundcloud.example/t")], engine):
        res = play_mod.play_via_registry(
            "Song", search_fn=search_fn, download_fn=download_fn,
            registry=reg)
    assert res.status == "ready"
    assert res.source_id == "soundcloud"
    desc = bridge.music_source_desc(reg, "soundcloud")
    assert reg.stats_for(desc)["ok"] >= 1


# ── the full-track verdict is three-valued ────────────────────────────


def test_full_track_verdict_three_states(tmp_path):
    from nomorals.media.library import full_track_verdict

    path = _audio_file(tmp_path)
    with mock.patch("nomorals.media.library.read_metadata",
                    return_value={"duration": 290.0}):
        state, _why = full_track_verdict(path, 300.0)
        assert state == "verified"
    with mock.patch("nomorals.media.library.read_metadata",
                    return_value={"duration": 25.0}):
        state, why = full_track_verdict(path, 300.0)
        assert state == "rejected"
        assert "preview" in why
    # no expected duration → nothing to verify against: UNVERIFIED,
    # never silently "passed"
    state, _why = full_track_verdict(path, 0.0)
    assert state == "unverified"
    # expected known but the file itself unprobeable → UNVERIFIED
    with mock.patch("nomorals.media.library.read_metadata",
                    return_value={"duration": 0.0}):
        state, _why = full_track_verdict(path, 300.0)
        assert state == "unverified"


def test_send_in_chat_state_three_valued_no_proof_claim(tmp_path):
    """The handler's verdict is three-valued in effect (doc 7 §2):
    the state helper distinguishes verified / rejected / UNVERIFIED
    (yesterday's helper returned "" for both verified and
    couldn't-check, which read as proof).  An unverified download is
    delivered under its plain identity line — the line claims the
    track, never a verification — and a preview-length download is
    refused outright, never sent as the track."""
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    path = _audio_file(tmp_path)
    engine = _FakeEngine({"https://youtu.be/x": {
        "ok": True, "path": path, "title": "Song"}})
    rt = _runtime(chat=True)
    chat = SimpleNamespace(platform="telegram", chat_id="42")
    item = {"kind": "youtube", "path": "https://youtu.be/x",
            "title": "Song", "artist": "", "duration": 0.0}
    state = RuntimeMediaMixin._play_verify_full_track_state
    # nothing on record to check against → UNVERIFIED, not a pass
    assert state(path, item)[0] == "unverified"
    with mock.patch("nomorals.media.library.read_metadata",
                    return_value={"duration": 290.0}):
        assert state(path, dict(item, duration=300.0))[0] == "verified"
    with mock.patch("nomorals.media.library.read_metadata",
                    return_value={"duration": 25.0}):
        assert state(path, dict(item, duration=300.0))[0] == "rejected"
        # a preview-length arrival is refused, never delivered as
        # the song
        rt.gateway.send_file.reset_mock()
        text, sent = rt._play_send_in_chat_result(
            engine, dict(item, duration=300.0), chat)
        assert sent is False
        assert "preview" in text
        assert rt.gateway.send_file.call_count == 0
    # the unverified delivery goes out under the plain identity
    # line — no verification claim anywhere in it
    text, sent = rt._play_send_in_chat_result(engine, item, chat)
    assert sent is True
    assert text == "🎵 Song"
    assert "verif" not in text.lower()
