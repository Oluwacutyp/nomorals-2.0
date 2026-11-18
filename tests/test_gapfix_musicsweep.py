"""Lane 26 — music-chain sweep: per-source copies survive, dead links
are remembered, and a failed search is evidence, not silence.

The owner's correction (2026-11-17): the defect is at SOURCE level.
Three behaviours pinned here:
(a) the sweep dedupes by URL, so the same song from SoundCloud,
    mdundo and bandcamp all survive for the fall-through;
(b) a download failure is recorded in the durable failure memory —
    the next sweep demotes that URL and the fall-through skips it
    without re-attempting, and a 403 / timeout / 5xx / site error on
    one copy skips the source's other copies for the same song;
(c) a search fetch that times out or 403s (netnaija / tooxclusive)
    is recorded as a search failure, so the sweep reports
    "failed: <kind>" instead of "no results".

Every test redirects stores with the app's own NM_HOME override
(never the real home, never the repo tree). No live network: the
search-fetch failures are raised through a patched urlopen.

Pins rule (#169): no bare-swallow handler anywhere in this file —
where a pin must swallow, it binds ``as exc`` + ``assert
isinstance(exc, Exception)``.
"""

import urllib.error
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media import download_failures as df
from nomorals.media.resolver import ResolvedAudio, SourceResolver
from nomorals.media.sources import NetNaijaSource, TooXclusiveSource

_RT_LOGGER = "nomorals.agents.partner.runtime_media"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    df.clear_failures()
    yield
    df.clear_failures()


# ── failure-kind taxonomy + URL normalisation ─────────────────────────

def test_classify_failure_kinds():
    assert df.classify_failure(TimeoutError("timed out")) == "timeout"
    # socket.timeout IS TimeoutError (alias) — the line above pins it.
    assert df.classify_failure("HTTP Error 403: Forbidden") == "http_403"
    assert df.classify_failure(
        urllib.error.HTTPError("http://x/", 403, "Forbidden", {}, None)
    ) == "http_403"
    assert df.classify_failure(
        urllib.error.HTTPError("http://x/", 404, "Not Found", {}, None)
    ) == "http_404"
    assert df.classify_failure(
        urllib.error.HTTPError("http://x/", 503, "Unavailable", {}, None)
    ) == "http_5xx"
    assert df.classify_failure("server error 500") == "http_5xx"
    assert df.classify_failure(
        urllib.error.URLError("no route to host")) == "site_error"
    assert df.classify_failure("file not found") == "http_404"
    assert df.classify_failure("some other breakage") == "download_error"
    assert df.classify_failure("") == "download_error"


def test_normalize_url():
    assert df.normalize_url("HTTPS://Example.COM/Path/") == \
        "https://example.com/Path"
    assert df.normalize_url("https://example.com/a#frag") == \
        "https://example.com/a"
    assert df.normalize_url("https://example.com/a?sig=1") == \
        "https://example.com/a?sig=1"
    assert df.normalize_url("/home/x/song.mp3") == "/home/x/song.mp3"
    assert df.normalize_url("") == ""
    assert df.normalize_url(None) == ""


# ── the memory itself ─────────────────────────────────────────────────

def test_url_failure_recorded_and_dead():
    df.record_url_failure("https://sc.example/track", "http_403",
                          source="soundcloud")
    assert df.is_url_dead("https://sc.example/track") is True
    assert df.dead_kind_for_url("https://sc.example/track") == "http_403"
    assert df.is_url_dead("https://other.example/track") is False
    assert df.dead_kind_for_url("https://other.example/track") is None
    # The trailing slash is the same URL.
    assert df.is_url_dead("https://sc.example/track/") is True


def test_url_failure_window_expires():
    df.record_url_failure("https://sc.example/track", "http_403")
    assert df.is_url_dead("https://sc.example/track", within_s=3600) is True
    assert df.is_url_dead("https://sc.example/track", within_s=0) is False


def test_search_failure_recorded_and_read():
    assert df.last_search_failure("netnaija") is None
    df.record_search_failure("netnaija", "timeout")
    assert df.last_search_failure("netnaija") == "timeout"
    assert df.last_search_failure("tooxclusive") is None
    # A search failure is not a URL failure.
    assert df.is_url_dead("netnaija") is False


def test_memory_persists_to_disk(tmp_path):
    import json

    df.record_url_failure("https://sc.example/track", "timeout",
                          source="soundcloud")
    store_path = (tmp_path / "home" / "failures"
                  / "media-download-failures.json")
    assert store_path.exists()
    doc = json.loads(store_path.read_text(encoding="utf-8"))
    assert doc["entries"], "nothing persisted"
    row = doc["entries"][-1]
    assert row["site"] == "https://sc.example/track"
    assert row["exc_type"] == "timeout"


def test_ledger_reloads_when_home_moves(tmp_path, monkeypatch):
    # The ledger used to load once per process, so a test that moved
    # NM_HOME kept answering from the first home's entries — the
    # fall-through tests' fixture URLs, recorded dead by one test,
    # were then skipped by the next test sharing the home. The
    # memory now follows the current home (and writes through to
    # it): dead under home A, unknown under a fresh home B, dead
    # again back under A.
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    monkeypatch.setenv("NM_HOME", str(home_a))
    df.record_url_failure("https://sc.example/track", "http_403",
                          source="soundcloud")
    assert df.dead_kind_for_url("https://sc.example/track") == "http_403"
    monkeypatch.setenv("NM_HOME", str(home_b))
    assert df.dead_kind_for_url("https://sc.example/track") is None
    monkeypatch.setenv("NM_HOME", str(home_a))
    assert df.dead_kind_for_url("https://sc.example/track") == "http_403"


def test_expired_url_failure_is_not_dead():
    df.record_url_failure("https://sc.example/track", "http_403")
    # Backdate the entry past the TTL (test-only white-box poke —
    # the only way to age a record without waiting six hours).
    df._ledger[0]["ts"] -= 7 * 3600
    assert df.is_url_dead("https://sc.example/track") is False


# ── sweep: per-source copies survive ──────────────────────────────────

_ALL_MANY = ("_netnaija_search_many", "_notjustok_search_many",
             "_tooxclusive_search_many", "_mp3pm_search_many",
             "_mp3juice_search_many", "_soundcloud_search_many",
             "_bandcamp_search_many", "_youtube_search_many",
             "_youtube_music_search_many", "_mdundo_search_many",
             "_archiveorg_search_many", "_jamendo_search_many",
             "_fma_search_many", "_jiosaavn_search_many",
             "_musicbrainz_search_many", "_openverse_search_many",
             "_wikimedia_search_many", "_itunes_search_many",
             "_deezer_search_many")


def _cand(kind, url, title="Tell Me One Lie", artist="Ronis Goliath"):
    return ResolvedAudio(ok=True, path_or_url=url, title=title,
                         artist=artist, kind=kind,
                         source_name=f"{kind} search")


def _sweep(resolver, **overrides):
    """Run the detailed sweep with every source mocked to [] except
    the named overrides (an unmocked source runs live — never)."""
    patches = []
    try:
        for name in _ALL_MANY:
            patches.append(mock.patch.object(
                resolver, name,
                return_value=overrides.get(name, [])))
        for p in patches:
            p.start()
        return resolver.search_candidates_detailed("tell me one lie",
                                                   limit=8)
    finally:
        for p in patches:
            p.stop()


def test_same_song_per_source_copies_survive():
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    cands, _attempts = _sweep(
        r,
        _soundcloud_search_many=[_cand("soundcloud",
                                       "https://soundcloud.com/r/tell-me-one-lie")],
        _bandcamp_search_many=[_cand("bandcamp",
                                     "https://bc.example/track/tell-me-one-lie")],
        _mdundo_search_many=[_cand("mdundo",
                                   "https://mdundo.com/song/tell-me-one-lie")])
    assert [c.kind for c in cands] == ["soundcloud", "bandcamp", "mdundo"]
    assert len({c.path_or_url for c in cands}) == 3


def test_same_url_from_two_sources_collapses():
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    shared = "https://cdn.example/song.mp3"
    cands, _attempts = _sweep(
        r,
        _soundcloud_search_many=[_cand("soundcloud", shared)],
        _bandcamp_search_many=[_cand("bandcamp", shared)])
    assert len(cands) == 1
    assert cands[0].kind == "soundcloud"


def test_same_url_trailing_slash_collapses():
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    cands, _attempts = _sweep(
        r,
        _soundcloud_search_many=[_cand(
            "soundcloud", "https://soundcloud.com/r/tell-me-one-lie")],
        _bandcamp_search_many=[_cand(
            "bandcamp", "https://soundcloud.com/r/tell-me-one-lie/")])
    assert len(cands) == 1


def test_dead_url_demoted_behind_live_copy():
    sc_url = "https://soundcloud.com/r/tell-me-one-lie"
    df.record_url_failure(sc_url, "http_403", source="soundcloud")
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    cands, _attempts = _sweep(
        r,
        _soundcloud_search_many=[_cand("soundcloud", sc_url)],
        _mdundo_search_many=[_cand(
            "mdundo", "https://mdundo.com/song/tell-me-one-lie")])
    assert [c.kind for c in cands] == ["mdundo", "soundcloud"]


def test_expired_failure_not_demoted():
    sc_url = "https://soundcloud.com/r/tell-me-one-lie"
    df.record_url_failure(sc_url, "http_403", source="soundcloud")
    df._ledger[0]["ts"] -= 7 * 3600
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    cands, _attempts = _sweep(
        r,
        _soundcloud_search_many=[_cand("soundcloud", sc_url)],
        _mdundo_search_many=[_cand(
            "mdundo", "https://mdundo.com/song/tell-me-one-lie")])
    assert [c.kind for c in cands] == ["soundcloud", "mdundo"]


# ── sweep: a failed search is evidence, not "no results" ──────────────

def _raise_timeout(*args, **kwargs):
    raise TimeoutError("timed out")


def _raise_403(*args, **kwargs):
    raise urllib.error.HTTPError("https://tooxclusive.com/?s=x", 403,
                                 "Forbidden", {}, None)


class _FakeResp:
    """A urlopen response serving one fixed HTML page."""

    def __init__(self, html):
        self._html = html.encode("utf-8")
        self.headers = SimpleNamespace(get_content_charset=lambda: None)

    def read(self):
        return self._html

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_netnaija_timeout_is_search_failure(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _raise_timeout)
    assert NetNaijaSource().search("tell me one lie") == []
    assert df.last_search_failure("netnaija") == "timeout"


def test_tooxclusive_403_is_search_failure(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _raise_403)
    assert TooXclusiveSource().search("tell me one lie") == []
    assert df.last_search_failure("tooxclusive") == "http_403"


def test_netnaija_timeout_reported_failed_in_sweep(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _raise_timeout)
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    patches = []
    try:
        for name in _ALL_MANY:
            if name in ("_netnaija_search_many",):
                continue
            patches.append(mock.patch.object(r, name, return_value=[]))
        for p in patches:
            p.start()
        _cands, attempts = r.search_candidates_detailed("tell me one lie",
                                                        limit=8)
    finally:
        for p in patches:
            p.stop()
    joined = " | ".join(attempts)
    assert "netnaija-search — failed: timeout" in joined, joined


def test_tooxclusive_403_reported_failed_in_sweep(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _raise_403)
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    patches = []
    try:
        for name in _ALL_MANY:
            if name in ("_tooxclusive_search_many",):
                continue
            patches.append(mock.patch.object(r, name, return_value=[]))
        for p in patches:
            p.start()
        _cands, attempts = r.search_candidates_detailed("tell me one lie",
                                                        limit=8)
    finally:
        for p in patches:
            p.stop()
    joined = " | ".join(attempts)
    assert "tooxclusive-search — failed: http_403" in joined, joined


def test_clean_empty_search_stays_no_result(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResp("<html></html>"))
    assert NetNaijaSource().search("tell me one lie") == []
    assert df.last_search_failure("netnaija") is None
    r = SourceResolver(SimpleNamespace(), soundcloud=None, spotify=None)
    _cands, attempts = _sweep(r)
    joined = " | ".join(attempts)
    assert "netnaija-search — no results" in joined, joined


# ── fall-through: dead links are skipped, not hammered ────────────────

class _FakeGateway:
    """Records sends/files; plays the adapter for edit verbs."""

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


def _item(kind, path, title="Tell Me One Lie"):
    return {"kind": kind, "path": path, "title": title,
            "artist": "Ronis Goliath"}


SC_URL = "https://soundcloud.com/ronis-goliath/tell-me-one-lie"
SC2_URL = "https://soundcloud.com/other-user/tell-me-one-lie"
MD_URL = "https://mdundo.com/song/tell-me-one-lie"


def test_fallthrough_403_skips_sibling_soundcloud_copies():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "HTTP Error 403: Forbidden"},
        {"ok": True, "path": "/tmp/md-song.mp3",
         "title": "Tell Me One Lie"},
    ]
    out, sent = rt._play_send_with_fallthrough(
        engine, _item("soundcloud", SC_URL),
        [_item("soundcloud", SC2_URL), _item("mdundo", MD_URL)],
        [], chat)
    # The second SoundCloud copy was never attempted — the 403
    # condemned the source for this song — mdundo landed instead.
    assert engine.download.call_count == 2
    assert engine.download.call_args_list[0][0][0]["kind"] == "soundcloud"
    assert engine.download.call_args_list[1][0][0]["kind"] == "mdundo"
    assert sent is True
    assert len(rt.gateway.files) == 1
    # And the dead link is remembered for the next /play.
    assert df.dead_kind_for_url(SC_URL) == "http_403"
    assert df.is_url_dead(SC2_URL) is False


def test_fallthrough_skips_already_dead_url():
    df.record_url_failure(MD_URL, "http_403", source="mdundo")
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": True, "path": "/tmp/sc-song.mp3", "title": "Tell Me One Lie"}
    out, sent = rt._play_send_with_fallthrough(
        engine, _item("mdundo", MD_URL),
        [_item("soundcloud", SC_URL)], [], chat)
    assert engine.download.call_count == 1
    assert engine.download.call_args_list[0][0][0]["kind"] == "soundcloud"
    assert sent is True


def test_fallthrough_generic_error_does_not_skip_siblings():
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "connection reset by peer"},
        {"ok": True, "path": "/tmp/sc2-song.mp3",
         "title": "Tell Me One Lie"},
    ]
    out, sent = rt._play_send_with_fallthrough(
        engine, _item("soundcloud", SC_URL),
        [_item("soundcloud", SC2_URL)], [], chat)
    # A generic error condemns only its own URL — the sibling copy
    # is still attempted.
    assert engine.download.call_count == 2
    assert engine.download.call_args_list[1][0][0]["path"] == SC2_URL
    assert sent is True
    assert df.dead_kind_for_url(SC_URL) == "download_error"


def test_fallthrough_wrong_file_is_remembered(monkeypatch):
    from nomorals.media import download_orchestrator as do

    def _verify(self, path, kind, expected_duration=0.0):
        if path == "/tmp/sc-song.mp3":
            return SimpleNamespace(ok=False,
                                   facts={"verify_state": "rejected"},
                                   reason="preview-length clip")
        return SimpleNamespace(ok=True, facts={}, reason="")

    monkeypatch.setattr(do.DownloadOrchestrator, "verify", _verify)
    rt = _handler()
    chat = _chat()
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": True, "path": "/tmp/sc-song.mp3",
         "title": "Tell Me One Lie"},
        {"ok": True, "path": "/tmp/md-song.mp3",
         "title": "Tell Me One Lie"},
    ]
    out, sent = rt._play_send_with_fallthrough(
        engine, _item("soundcloud", SC_URL),
        [_item("mdundo", MD_URL)], [], chat)
    assert engine.download.call_count == 2
    assert sent is True
    assert df.dead_kind_for_url(SC_URL) == "wrong_file"
