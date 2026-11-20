"""Mdundo demotion + download-failure fix (owner live report 2026-11-18).

The screenshot: ``/play armed and dangerous`` where EVERY candidate
was mdundo — the direct leg died with "unable to open for writing:
[Errno 2] No such file or directory", the failure was classified as a
site error, and that condemned every mdundo copy for the song
("mdundo failed on this song" x2) while mp3.pm never surfaced.

Rules under test:
- (1) mdundo sits last among the audio-capable sources in EVERY order
  list (MUSIC_SEARCH_SOURCES, SOURCE_CHAIN, the sweep fns, the chain
  strategies) and the learned ordering cannot float it back up.
- (2) when mp3.pm and mdundo both confidently match, the attempt
  order leads with mp3.pm (source order is the tiebreak).
- (3a) a missing parent directory on the download write path is
  created and the download retried once, on both yt-dlp backends.
- (3b) a LOCAL write failure is not a site error: it never condemns
  the source kind, so the next copy is still tried.
- (4) mp3.pm parses this query shape (Mobb Deep "Armed And
  Dangerous") from its server-rendered search page.

All fetches / downloads are stubbed — no network, no live download.
"""

import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media import download_failures as _df
from nomorals.media.download_failures import SOURCE_DEAD_KINDS
from nomorals.media.download_failures import (
    classify_failure as df_classify_failure,
)
from nomorals.media.downloader import FailureKind, classify_failure
from nomorals.media.resolver import ResolvedAudio, SourceResolver
from nomorals.media.source_registry import SourceRegistry
from nomorals.media.sources import (
    MUSIC_SEARCH_SOURCES,
    SOURCE_CHAIN,
)
from nomorals.tools.media import download as media_download

#: The exact failure text from the owner's screenshot (truncated
#: there at "No such file or d" — the full yt-dlp text is restored).
WRITE_ENOENT = (
    "download failed: 1. direct — site_error: download failed: "
    "ERROR: unable to open for writing: [Errno 2] No such file or "
    "directory: 'C:\\Users\\Administrator\\.nomorals\\workspace\\media"
    "\\song_get [\\u29f8e=1763489000&h=0123456789abcdef&t="
    "Bulldog-Armed-Dangerous].mp3.part'"
)

#: Audio-capable sources mdundo must sit below (owner order
#: 2026-11-18).  Metadata-only sources (musicbrainz / openverse /
#: wikimedia / itunes) keep their relative order AFTER mdundo.
AUDIO_SOURCES = (
    "mp3pm-search", "mp3juice-search", "soundcloud-search",
    "bandcamp-search", "youtube-search", "youtube-music-search",
    "archiveorg-search", "jamendo-search", "fma-search",
    "jiosaavn-search", "boomplay-search", "deezer-search",
    "spotify-search",
)


@pytest.fixture(autouse=True)
def _isolated_download_failure_home(tmp_path, monkeypatch):
    # The fall-through consults the durable download-failure memory,
    # so tests that reuse fixture URLs must not share one home.
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    _df.clear_failures()
    yield
    _df.clear_failures()


# ── (1) the order ────────────────────────────────────────────────────


class TestMdundoOrder(unittest.TestCase):
    def test_music_search_sources_mdundo_last_audio(self):
        order = list(MUSIC_SEARCH_SOURCES)
        md = order.index("mdundo-search")
        for name in AUDIO_SOURCES:
            self.assertLess(order.index(name), md, name)
        # Metadata-only sources keep their relative order AFTER mdundo.
        self.assertLess(md, order.index("musicbrainz-search"))
        self.assertLess(order.index("musicbrainz-search"),
                        order.index("openverse-search"))
        self.assertLess(order.index("openverse-search"),
                        order.index("wikimedia-search"))
        self.assertLess(order.index("wikimedia-search"),
                        order.index("itunes-search"))

    def test_source_chain_mdundo_after_deezer(self):
        names = [name for name, _src in SOURCE_CHAIN]
        md = names.index("mdundo-search")
        self.assertLess(names.index("mp3pm-search"), md)
        self.assertLess(names.index("deezer-search"), md)
        self.assertLess(md, names.index("musicbrainz-search"))

    def test_sweep_calls_mdundo_after_audio_sources(self):
        # The detailed sweep calls every source fn in list order;
        # stub them all to record the order they are tried in.
        resolver = SourceResolver()
        called: list[str] = []
        fns = (
            "netnaija", "notjustok", "tooxclusive", "mp3pm",
            "mp3juice", "soundcloud", "bandcamp", "youtube",
            "youtube-music", "mdundo", "archiveorg", "jamendo",
            "fma", "jiosaavn", "musicbrainz", "openverse",
            "wikimedia", "itunes", "deezer",
        )
        for key in fns:
            def _stub(q, limit, _key=key):
                called.append(f"{_key}-search")
                return []
            setattr(resolver, f"_{key.replace('-', '_')}_search_many",
                    _stub)
        out, _attempts = resolver.search_candidates_detailed(
            "armed and dangerous", limit=25)
        self.assertEqual(out, [])
        # Owner order 2026-11-19: mp3.pm AND the MP3Juice mirror
        # (service-et-reception.fr) sit at the top of the sweep,
        # before the regional blogs; mdundo stays last among the
        # audio-capable sources.
        self.assertEqual(called, [
            "mp3pm-search", "mp3juice-search",
            "netnaija-search", "notjustok-search",
            "tooxclusive-search", "soundcloud-search",
            "bandcamp-search", "youtube-search",
            "youtube-music-search", "archiveorg-search",
            "jamendo-search", "fma-search", "jiosaavn-search",
            "deezer-search", "mdundo-search", "musicbrainz-search",
            "openverse-search", "wikimedia-search", "itunes-search",
        ])

    def test_chain_strategies_mdundo_after_spotify(self):
        # The first-hit chain tries each strategy in list order;
        # stub them all to record the order (all return no result).
        resolver = SourceResolver()
        called: list[str] = []

        def _blog(q, cls_name, key):
            called.append(f"{key}-search")
            return ResolvedAudio()

        def _meta(q, cls_name, key, note=""):
            called.append(f"{key}-search")
            return ResolvedAudio()

        def _preview(q, cls_name, key):
            called.append(f"{key}-search")
            return ResolvedAudio()

        def _single(name):
            def _fn(q):
                called.append(f"{name}-search")
                return ResolvedAudio()
            return _fn

        resolver._s_blog_search = _blog  # noqa: SLF001
        resolver._s_metadata_search = _meta  # noqa: SLF001
        resolver._s_preview_search = _preview  # noqa: SLF001
        for name in ("soundcloud", "bandcamp", "youtube",
                     "youtube-music", "archiveorg", "boomplay",
                     "spotify"):
            setattr(resolver, f"_s_{name.replace('-', '_')}_search",
                    _single(name))
        result = resolver._resolve_text(  # noqa: SLF001
            "armed and dangerous", ResolvedAudio())
        self.assertFalse(result.ok)
        md = called.index("mdundo-search")
        for name in AUDIO_SOURCES:
            self.assertLess(called.index(name), md, name)
        self.assertLess(md, called.index("musicbrainz-search"))

    def test_learned_order_cannot_float_mdundo_up(self):
        # EC2's registry has learned state: mdundo served 20 times,
        # mp3.pm never — on base the score sort floats mdundo first.
        # The owner's demotion must hold anyway.  Descriptors are
        # built from the seed table's rows (a bare registry carries
        # no seeds; DEFAULT_REGISTRY is shared and must not be
        # mutated by a test).
        from nomorals.media.source_registry import (
            Licence,
            SourceDescriptor,
            TrustTier,
        )

        reg = SourceRegistry()
        mdundo = SourceDescriptor(
            nation="KE", language="en", domain="play.mdundo.com",
            name="Mdundo", kinds=("music", "music-search"),
            licence=Licence.LICENSED, trust_tier=TrustTier.COMMUNITY)
        mp3pm = SourceDescriptor(
            nation="GLOBAL", language="en", domain="mp3.pm",
            name="MP3.pm", kinds=("music", "music-search"),
            licence=Licence.UNLICENSED,
            trust_tier=TrustTier.UNVERIFIED)
        reg.register(mdundo)
        reg.register(mp3pm)
        now = time.time()
        for _ in range(20):
            reg.record_outcome(mdundo, True, latency_s=0.5,
                               fetched_at=now)
        resolver = SourceResolver(source_registry=reg)
        items = [
            ("mp3pm-search", object(), mp3pm),
            ("mdundo-search", object(), mdundo),
        ]
        ordered = resolver._order_music(reg, items)  # noqa: SLF001
        self.assertEqual([name for name, _fn, _d in ordered],
                         ["mp3pm-search", "mdundo-search"])


# ── (2) the pick list ────────────────────────────────────────────────


class _FakeGateway:
    """Records sends/files; plays the adapter for the fall-through."""

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


def _item(kind, path, title, artist):
    return {"kind": kind, "path": path, "title": title, "artist": artist}


_MP3PM_ITEM = _item(
    "mp3pm",
    "https://28682549-mobb-deep.mp3.pm/song/1-armed-and-dangerous/",
    "Armed And Dangerous", "Mobb Deep")
_MDUNDO_ITEM = _item(
    "mdundo", "https://mdundo.com/stream/1571564",
    "Armed And Dangerous", "Bulldog")


class TestPicklistTiebreak(unittest.TestCase):
    def test_gated_tiebreak_mp3pm_before_mdundo(self):
        # Both confidently match with the SAME score (title F1 1.0)
        # and mdundo arrives first — the chain order must break the
        # tie toward the real source, not the demoted one.
        from nomorals.media.picklist import match_confidence

        mp3_score = match_confidence(
            "Armed And Dangerous", "Mobb Deep",
            "armed and dangerous", "")
        md_score = match_confidence(
            "Armed And Dangerous", "Bulldog",
            "armed and dangerous", "")
        self.assertEqual(mp3_score, md_score)
        self.assertEqual(mp3_score, 1.0)
        rt = _handler()
        engine = mock.Mock()
        engine.download.return_value = {
            "ok": True, "path": "/tmp/mobb-deep.mp3",
            "title": "Armed And Dangerous"}
        out, sent = rt._play_send_with_fallthrough(
            engine, _MDUNDO_ITEM, [_MP3PM_ITEM], [], _chat(),
            match=("armed and dangerous", ""))
        self.assertTrue(sent, out)
        first = engine.download.call_args_list[0][0][0]
        self.assertEqual(first["kind"], "mp3pm", first)
        self.assertIn("/tmp/mobb-deep.mp3", rt.gateway.files[0])

    def test_local_write_failure_does_not_condemn_kind(self):
        # The Bulldog copy fails with the screenshot's local-write
        # error; the Lucid Vybz copy (same kind) must still be tried.
        rt = _handler()
        engine = mock.Mock()
        lucid = _item("mdundo", "https://mdundo.com/stream/1571565",
                      "Armed And Dangerous", "Lucid Vybz")
        engine.download.side_effect = [
            {"ok": False, "reason": WRITE_ENOENT},
            {"ok": True, "path": "/tmp/lucid.mp3",
             "title": "Armed And Dangerous"},
        ]
        out, sent = rt._play_send_with_fallthrough(
            engine, _MDUNDO_ITEM, [lucid], [], _chat(),
            match=("armed and dangerous", ""))
        self.assertTrue(sent, out)
        self.assertEqual(engine.download.call_count, 2)
        second = engine.download.call_args_list[1][0][0]
        self.assertEqual(second["kind"], "mdundo", second)


# ── (3b) the classification ──────────────────────────────────────────


class TestLocalWriteClassification(unittest.TestCase):
    def test_downloader_classifies_local_write(self):
        kind = classify_failure(
            "download failed: ERROR: unable to open for writing: "
            "[Errno 2] No such file or directory: '/x/song.mp3.part'")
        self.assertEqual(kind, FailureKind.LOCAL_WRITE_ERROR)

    def test_failure_memory_classifies_local_write(self):
        kind = df_classify_failure(WRITE_ENOENT)
        self.assertEqual(kind, "local_error")
        self.assertNotIn(kind, SOURCE_DEAD_KINDS)

    def test_oserror_local_errno_is_local(self):
        err = OSError(2, "No such file or directory")
        self.assertEqual(df_classify_failure(err), "local_error")
        err = OSError(28, "No space left on device")
        self.assertEqual(df_classify_failure(err), "local_error")


# ── (3a) the mkdir retry ─────────────────────────────────────────────


class _FakeYtDlpModule:
    """yt-dlp stand-in: first extraction dies with the screenshot's
    ENOENT (its prepared filename lives in a subdir that does not
    exist); the second lands the file there."""

    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.calls = 0
        self.sub = target_dir / "sub"
        self.dest = self.sub / "song_get [x].mp3"

    class _DL:
        def __init__(self, outer, opts):
            self._outer = outer

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            outer = self._outer
            outer.calls += 1
            if outer.calls == 1:
                raise Exception(
                    "ERROR: unable to open for writing: [Errno 2] "
                    f"No such file or directory: '{outer.dest}.part'")
            outer.dest.parent.mkdir(parents=True, exist_ok=True)
            outer.dest.write_bytes(b"\xff\xfb" + b"\x00" * 100)
            return {"id": "song_get", "title": "song_get",
                    "filepath": str(outer.dest), "ext": "mp3"}

        def prepare_filename(self, info):
            return info["filepath"]

    def YoutubeDL(self, opts):  # noqa: N802 - yt-dlp's own name
        return self._DL(self, opts)


class TestDownloadMkdirRetry(unittest.TestCase):
    def test_python_api_retries_missing_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fresh-target"
            fake = _FakeYtDlpModule(target)
            with mock.patch("nomorals.tools.media.load_optional",
                            return_value=fake):
                result = media_download(
                    "https://mdundo.com/stream/1571564", target,
                    audio_only=True)
            self.assertEqual(fake.calls, 2)
            self.assertTrue(Path(result["path"]).is_file())
            self.assertEqual(Path(result["path"]).parent, target / "sub")

    def test_cli_retries_missing_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fresh-target"
            target.mkdir(parents=True)
            sub = target / "sub"
            dest = sub / "song_get [x].mp3"
            calls = {"n": 0}

            class _FakeProc:
                def __init__(self, argv, **kw):
                    calls["n"] += 1
                    self._n = calls["n"]

                def communicate(self, timeout=None):
                    if self._n == 1:
                        return (
                            f"[download] Destination: {dest}\n",
                            "ERROR: unable to open for writing: "
                            "[Errno 2] No such file or directory: "
                            f"'{dest}.part'\n")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b"\xff\xfb" + b"\x00" * 100)
                    return (f"[download] Destination: {dest}\n", "")

                @property
                def returncode(self):
                    return 1 if self._n == 1 else 0

            def _which(cmd):
                return "/usr/bin/yt-dlp" if cmd == "yt-dlp" else None

            with mock.patch("nomorals.tools.media.load_optional",
                            return_value=None), \
                mock.patch("nomorals.tools.media.which",
                           side_effect=_which), \
                mock.patch.object(subprocess, "Popen", _FakeProc):
                result = media_download(
                    "https://mdundo.com/stream/1571564", target,
                    audio_only=True)
            self.assertEqual(calls["n"], 2)
            self.assertTrue(Path(result["path"]).is_file())


# ── (4) mp3.pm for this query ────────────────────────────────────────

#: Row modeled on the live search page for "armed and dangerous"
#: (2026-11-18, first result Mobb Deep "Armed And Dangerous"), in the
#: exact <li class="cplayer-sound-item"> shape of the recorded
#: 2026-11-17 row for "tell me one lie".
_MP3PM_ARMED_HTML = (
    '<ul class="mp3list">'
    '<li class="cplayer-sound-item" data-sound-id="248463782" '
    'data-sound-url="https://cs1.mp3.pm/listen/248463782/TOKEN/'
    'Mobb_Deep_-_Armed_And_Dangerous_(mp3.pm).mp3" '
    'data-download-url="https://cs1.mp3.pm/download/248463782/TOKEN/'
    'Mobb_Deep_-_Armed_And_Dangerous_(mp3.pm).mp3" '
    'data-share-url="https://mp3.pmhttps://28682549-mobb-deep.mp3.pm/'
    'song/248463782-armed-and-dangerous/">'
    '<h4><a href="https://28682549-mobb-deep.mp3.pm/">'
    '<i class="cplayer-data-sound-author">Mobb Deep</i></a> '
    '<a href="https://28682549-mobb-deep.mp3.pm/song/'
    '248463782-armed-and-dangerous/">'
    '<b class="cplayer-data-sound-title">Armed And Dangerous</b></a>'
    '</h4><em class="cplayer-data-sound-time">03:22</em></li>'
    '</ul>'
)


class TestMp3pmArmedAndDangerous(unittest.TestCase):
    def test_search_parses_armed_and_dangerous(self):
        from nomorals.media.sources import MP3PMSource

        src = MP3PMSource()
        src._get = lambda url: _MP3PM_ARMED_HTML  # noqa: SLF001
        found = src.search("armed and dangerous", limit=5)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].title, "Armed And Dangerous")
        self.assertEqual(found[0].artist, "Mobb Deep")
        self.assertEqual(found[0].source, "mp3pm")
        self.assertTrue(found[0].direct_url.startswith(
            "https://cs1.mp3.pm/listen/248463782/"))
        self.assertTrue(found[0].downloadable)
