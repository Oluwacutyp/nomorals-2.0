"""Music-chain fix — the owner's live complaint (lane 66).

Owner words (2026-11-16/17/18, main chat, quoted exactly):
- "Move mdundo down, after the big globals"
- "Mundo or whatever is not a source it is crap... I said move it
  down!!!! And let real shits run like np3.pm and the likes"
- "Same for YouTube multisearch" (YouTube legs dying on the same
  song — the bot check)
- "Tf is this, keeps hammering soundcloud" (the same SoundCloud
  song retrying the same link over and over)

The defects, pinned here:
1. The EFFECTIVE order is not the listed order: ``_order_music``
   re-sorted actives by learned registry score, so mp3.pm sat
   FOURTH in the sweep the code actually runs.  mp3.pm is now
   PINNED first (``PINNED_CHAIN_SOURCES``) and mdundo stays DEMOTED
   last among audio sources — learned scores cannot move either.
2. mp3.pm's search route: ``/api/search?q=`` 404s and never existed
   as a JSON endpoint; the live page ``/search?q=`` is the source
   of truth (probed 2026-11-20: 50 cplayer rows, each with a direct
   listen + download MP3).
3. The download engine never consulted the dead-URL memory and
   never recorded into it — a dead link was re-run through the
   whole chain on every call, and a link proved dead mid-chain was
   re-fetched by the later rungs.  Now: the entry guard refuses a
   remembered-dead URL fast; a stage that proves the link dead
   stops the chain; exhaustion records the URL.
4. YouTube: one /play invoked yt-dlp up to 5x into the same bot
   wall, each leg reporting "no results".  Now: a recorded
   ``bot_block`` gates BOTH legs (one wall), and the first bot
   detection records it — the sweep reports "failed: bot_block".
5. service-et-reception.fr (owner, 2026-11-15 — "Music
   source....", "can't we rea it?", "mp3.pm should be at the
   top of source for music same with this one"): an MP3Juice
   clone holding no catalogue of its own (REA 2026-11-16) —
   POST search (field ``query``) returning rows whose Download
   buttons embed the YouTube id.  It sits at the top WITH
   mp3.pm, for typed song names as well as pasted links.  Its
   converter API answers HTTP 403 from here and its iframe
   host is unreachable from here (both walled, probed
   2026-11-20) — so the wired download is the YouTube watch
   URL via yt-dlp, not a guessed converter route.
"""

import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nomorals.core.errors import ToolError
from nomorals.media import download_failures as _df
from nomorals.media.downloader import (
    FailureKind,
    MediaDownloader,
    StageLog,
)
from nomorals.media.playback import PlaybackEngine
from nomorals.media.resolver import ResolvedAudio, SourceResolver
from nomorals.media.source_registry import (
    Licence,
    SourceDescriptor,
    SourceRegistry,
    TrustTier,
)
from nomorals.media.sources import (
    MUSIC_SEARCH_SOURCES,
    SOURCE_CHAIN,
    MP3JuiceSource,
    MP3PMSource,
    SourceCandidate,
)

#: The sweep / chain order the fix pins (mp3.pm first among remotes,
#: mdundo last among audio-capable sources — owner order 2026-11-18).
EXPECTED_SWEEP_ORDER = [
    "mp3pm-search", "mp3juice-search", "netnaija-search",
    "notjustok-search", "tooxclusive-search", "soundcloud-search",
    "bandcamp-search", "youtube-search",
    "youtube-music-search", "archiveorg-search", "jamendo-search",
    "fma-search", "jiosaavn-search", "deezer-search",
    "mdundo-search",
    "musicbrainz-search", "openverse-search", "wikimedia-search",
    "itunes-search",
]

#: Audio-capable sources mdundo must sit below (owner order
#: 2026-11-18).  Metadata-only sources keep their relative order
#: AFTER mdundo.
AUDIO_SOURCES = (
    "mp3pm-search", "mp3juice-search", "soundcloud-search",
    "bandcamp-search", "youtube-search", "youtube-music-search",
    "archiveorg-search", "jamendo-search", "fma-search",
    "jiosaavn-search", "boomplay-search", "deezer-search",
    "spotify-search",
)

#: Two cplayer rows in the shape the live mp3.pm search page
#: returns (probed 2026-11-20, /search?q=last+last): each row
#: carries a direct listen MP3 (data-sound-url), a direct download
#: MP3 (data-download-url), title, artist, duration and a song-page
#: link.
_MP3PM_PAGE = """<html><body><ul>
<li class="cplayer-sound-item" data-sound-url="https://cs1.mp3.pm/listen/111/last-last.mp3" data-download-url="https://cs1.mp3.pm/download/111/last-last.mp3">
  <span class="cplayer-data-sound-title">Last Last</span>
  <span class="cplayer-data-sound-author">Burna Boy</span>
  <span class="cplayer-data-sound-time">4:11</span>
  <a href="https://burna-boy.mp3.pm/song/111-last-last/">song</a>
</li>
<li class="cplayer-sound-item" data-sound-url="https://cs1.mp3.pm/listen/222/last-last-remix.mp3" data-download-url="https://cs1.mp3.pm/download/222/last-last-remix.mp3">
  <span class="cplayer-data-sound-title">Last Last (Remix)</span>
  <span class="cplayer-data-sound-author">Burna Boy</span>
  <span class="cplayer-data-sound-time">3:58</span>
  <a href="https://burna-boy.mp3.pm/song/222-last-last-remix/">song</a>
</li>
</ul></body></html>"""


@pytest.fixture(autouse=True)
def _isolated_download_failure_home(tmp_path, monkeypatch):
    # The dead-URL / search-failure memory is durable per NM_HOME,
    # so tests that record failures must not share one home.
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    _df.clear_failures()
    yield
    _df.clear_failures()


# ── (1) the order ────────────────────────────────────────────────────


class TestMusicChainOrder(unittest.TestCase):
    def test_listed_order_mp3pm_first_remote(self):
        order = list(MUSIC_SEARCH_SOURCES)
        self.assertEqual(order[0], "workspace-scan")
        self.assertEqual(order[1], "mp3pm-search")
        self.assertEqual(order[2], "mp3juice-search")
        mp3pm = order.index("mp3pm-search")
        for name in ("netnaija-search", "notjustok-search",
                     "tooxclusive-search"):
            self.assertLess(mp3pm, order.index(name), name)

    def test_listed_order_mdundo_last_audio(self):
        order = list(MUSIC_SEARCH_SOURCES)
        md = order.index("mdundo-search")
        for name in AUDIO_SOURCES:
            self.assertLess(order.index(name), md, name)
        self.assertLess(md, order.index("musicbrainz-search"))

    def test_source_chain_mp3pm_first_remote(self):
        names = [name for name, _src in SOURCE_CHAIN]
        self.assertLess(names.index("mp3pm-search"),
                        names.index("netnaija-search"))

    def test_sweep_effective_order_mp3pm_first(self):
        # The detailed sweep calls every source fn in list order;
        # stub them all to record the order they are tried in.
        resolver = SourceResolver()
        called: list[str] = []
        fns = (
            "mp3pm", "mp3juice", "netnaija", "notjustok",
            "tooxclusive", "soundcloud", "bandcamp",
            "youtube", "youtube-music", "archiveorg", "jamendo",
            "fma", "jiosaavn", "deezer",
            "mdundo", "musicbrainz", "openverse", "wikimedia",
            "itunes",
        )
        for key in fns:
            def _stub(q, limit, _key=key):
                called.append(f"{_key}-search")
                return []
            setattr(resolver, f"_{key.replace('-', '_')}_search_many",
                    _stub)
        out, _attempts = resolver.search_candidates_detailed(
            "last last", limit=25)
        self.assertEqual(out, [])
        self.assertEqual(called, EXPECTED_SWEEP_ORDER)

    def test_chain_effective_order_mp3pm_first(self):
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
        for name in ("netnaija", "soundcloud", "bandcamp", "youtube",
                     "youtube-music", "archiveorg", "boomplay",
                     "spotify"):
            setattr(resolver, f"_s_{name.replace('-', '_')}_search",
                    _single(name))
        result = resolver._resolve_text(  # noqa: SLF001
            "last last", ResolvedAudio())
        self.assertFalse(result.ok)
        self.assertEqual(called[0], "mp3pm-search")
        self.assertEqual(called[1], "mp3juice-search")
        md = called.index("mdundo-search")
        for name in AUDIO_SOURCES:
            self.assertLess(called.index(name), md, name)
        self.assertLess(md, called.index("musicbrainz-search"))

    def test_learned_scores_cannot_reorder_chain(self):
        # EC2's registry has learned state: netnaija, mp3juice and
        # mdundo served 20 times each, mp3.pm never — on the old
        # code the score sort floated netnaija first and mdundo
        # up.  The owner's order must hold anyway: mp3.pm pinned
        # first, mp3juice pinned second, mdundo demoted last among
        # actives.
        reg = SourceRegistry()
        mp3pm = SourceDescriptor(
            nation="GLOBAL", language="en", domain="mp3.pm",
            name="MP3.pm", kinds=("music", "music-search"),
            licence=Licence.UNLICENSED,
            trust_tier=TrustTier.UNVERIFIED)
        mp3juice = SourceDescriptor(
            nation="GLOBAL", language="en",
            domain="service-et-reception.fr",
            name="MP3Juice", kinds=("music", "music-search"),
            licence=Licence.UNLICENSED,
            trust_tier=TrustTier.UNVERIFIED)
        netnaija = SourceDescriptor(
            nation="NG", language="en", domain="netnaija.com",
            name="Netnaija", kinds=("music", "music-search"),
            licence=Licence.UNLICENSED,
            trust_tier=TrustTier.UNVERIFIED)
        mdundo = SourceDescriptor(
            nation="KE", language="en", domain="play.mdundo.com",
            name="Mdundo", kinds=("music", "music-search"),
            licence=Licence.LICENSED, trust_tier=TrustTier.COMMUNITY)
        for desc in (mp3pm, mp3juice, netnaija, mdundo):
            reg.register(desc)
        now = time.time()
        for _ in range(20):
            reg.record_outcome(mp3juice, True, latency_s=0.5,
                               fetched_at=now)
            reg.record_outcome(netnaija, True, latency_s=0.5,
                               fetched_at=now)
            reg.record_outcome(mdundo, True, latency_s=0.5,
                               fetched_at=now)
        resolver = SourceResolver(source_registry=reg)
        items = [
            ("netnaija-search", object(), netnaija),
            ("mdundo-search", object(), mdundo),
            ("mp3juice-search", object(), mp3juice),
            ("mp3pm-search", object(), mp3pm),
        ]
        ordered = resolver._order_music(reg, items)  # noqa: SLF001
        self.assertEqual([name for name, _fn, _d in ordered],
                         ["mp3pm-search", "mp3juice-search",
                          "netnaija-search", "mdundo-search"])


# ── (2) mp3.pm's route ───────────────────────────────────────────────


class TestMp3pmRoute(unittest.TestCase):
    def test_search_uses_page_route_not_api(self):
        seen: dict[str, str] = {}
        src = MP3PMSource()

        def _fake_get(url):
            seen["url"] = url
            return _MP3PM_PAGE
        src._get = _fake_get  # noqa: SLF001 - seam under test
        found = src.search("last last", limit=5)
        self.assertEqual(
            seen["url"], "https://mp3.pm/search?q=last+last")
        self.assertNotIn("api/search", seen["url"])
        self.assertEqual(len(found), 2)

    def test_parse_live_shape_rows(self):
        rows = MP3PMSource._parse_rows(_MP3PM_PAGE)  # noqa: SLF001
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "Last Last")
        self.assertEqual(rows[0]["artist"], "Burna Boy")
        self.assertEqual(rows[0]["duration"], 251.0)
        self.assertEqual(
            rows[0]["url"],
            "https://burna-boy.mp3.pm/song/111-last-last/")
        self.assertEqual(
            rows[0]["direct_url"],
            "https://cs1.mp3.pm/listen/111/last-last.mp3")
        self.assertEqual(
            rows[1]["direct_url"],
            "https://cs1.mp3.pm/listen/222/last-last-remix.mp3")

    def test_search_returns_candidates_with_direct_urls(self):
        src = MP3PMSource()
        src._get = lambda url: _MP3PM_PAGE  # noqa: SLF001
        found = src.search("last last", limit=5)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0].source, "mp3pm")
        self.assertEqual(found[0].title, "Last Last")
        self.assertEqual(found[0].artist, "Burna Boy")
        self.assertEqual(
            found[0].direct_url,
            "https://cs1.mp3.pm/listen/111/last-last.mp3")


#: Two result rows in the shape service-et-reception.fr's POST
#: search returns (REA 2026-11-16): each row carries an <h3>
#: title, a numeric data-id, and Download buttons embedding the
#: downloader-ytjar.online iframe with the YouTube video id.
#: The video ids below are real 11-char ids in the shape the
#: page carries (Rick Astley / Burna Boy official videos).
_MP3JUICE_PAGE = """<html><body>
<div class="result">
  <h3>Burna Boy - Last Last</h3>
  <div data-id="101"></div>
  <iframe src="https://downloader-ytjar.online/button.php?id=dQw4w9WgXcQ"></iframe>
  <iframe src="https://downloader-ytjar.online/button-2.php?id=dQw4w9WgXcQ"></iframe>
</div>
<div class="result">
  <h3>Burna Boy - Last Last (Remix)</h3>
  <div data-id="102"></div>
  <iframe src="https://downloader-ytjar.online/button.php?id=oQQ84L-b0Ss"></iframe>
</div>
</body></html>"""

#: Same rows WITHOUT the button iframes — the old numeric-only
#: shape (data-id + <h3>), for the fallback path.
_MP3JUICE_NUMERIC_PAGE = """<html><body>
<div class="result">
  <h3>Burna Boy - Last Last</h3>
  <div data-id="101"></div>
</div>
</body></html>"""


class TestMp3JuiceRoute(unittest.TestCase):
    def test_typed_query_posts_and_returns_youtube_candidates(self):
        # The owner's Nov 15 order, sharpened Nov 16: a TYPED
        # song name must reach this source's search route and come
        # back a real downloadable MP3 — pasted-links-only would
        # misunderstand "universal".  The query travels in the
        # POST body (field `query`); the results page's Download
        # buttons carry the YouTube id, and the candidate IS the
        # watch URL Devon's yt-dlp stage downloads.
        seen: dict[str, object] = {}
        src = MP3JuiceSource()

        def _fake_post(path, form):
            seen["path"] = path
            seen["form"] = dict(form)
            return _MP3JUICE_PAGE
        src._post = _fake_post  # noqa: SLF001 - seam under test
        found = src.search("last last", limit=5)
        self.assertEqual(seen["path"], "/")
        self.assertEqual(seen["form"], {"query": "last last"})
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0].source, "mp3juice")
        self.assertEqual(found[0].title, "Burna Boy - Last Last")
        self.assertEqual(
            found[0].url,
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(found[0].extra["yt_id"], "dQw4w9WgXcQ")
        self.assertEqual(
            found[1].url,
            "https://www.youtube.com/watch?v=oQQ84L-b0Ss")
        self.assertEqual(found[1].extra["yt_id"], "oQQ84L-b0Ss")

    def test_download_url_is_watch_url_for_youtube_rows(self):
        src = MP3JuiceSource()
        cand = SourceCandidate(
            source="mp3juice", title="Last Last", artist="",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            extra={"yt_id": "dQw4w9WgXcQ"})
        self.assertEqual(
            src.download_url(cand),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        # A candidate carrying only the watch URL in .url (no
        # extra) resolves the same way.
        cand2 = SourceCandidate(
            source="mp3juice", title="Last Last", artist="",
            url="https://www.youtube.com/watch?v=oQQ84L-b0Ss",
            extra={})
        self.assertEqual(
            src.download_url(cand2),
            "https://www.youtube.com/watch?v=oQQ84L-b0Ss")

    def test_pasted_youtube_link_needs_no_post(self):
        src = MP3JuiceSource()

        def _boom(path, form):
            raise AssertionError("a POST ran for a pasted link")
        src._post = _boom  # noqa: SLF001 - seam under test
        found = src.search(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", limit=5)
        self.assertEqual(len(found), 1)
        self.assertEqual(
            found[0].url,
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(found[0].extra["yt_id"], "dQw4w9WgXcQ")

    def test_numeric_rows_keep_old_shape(self):
        # No button iframes on the page: the numeric data-id rows
        # keep the old shape (site page + id), resolved by
        # download_url()'s endpoint guesses.
        src = MP3JuiceSource()
        src._post = lambda path, form: _MP3JUICE_NUMERIC_PAGE  # noqa: SLF001
        found = src.search("last last", limit=5)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].title, "Burna Boy - Last Last")
        self.assertEqual(
            found[0].url, "https://service-et-reception.fr/?id=101")
        self.assertEqual(found[0].extra["id"], "101")


# ── (3) the engine bound ─────────────────────────────────────────────


def _ctx(tmpdir):
    ctx = MagicMock()
    ctx.home = Path(tmpdir)
    ctx.db = MagicMock()
    ctx.settings = None
    return ctx


class TestDownloaderBound(unittest.TestCase):
    def test_dead_url_refused_fast_with_no_stage_run(self):
        # A URL the failure memory already knows is dead is
        # refused at the door — no stage runs, no fetch happens.
        url = "https://soundcloud.com/dead-artist/dead-track"
        _df.record_url_failure(url, "site_error")
        engine = MediaDownloader(_ctx(self._tmp), direct_retries=1)
        calls: list[str] = []

        def _boom(*args, **kwargs):
            calls.append("stage")
            raise AssertionError("a stage ran on a dead URL")
        engine._attempt_stage = _boom  # noqa: SLF001
        report = engine.download(url, dest_dir=self._tmp)
        self.assertFalse(report.ok)
        self.assertEqual(report.failure, FailureKind.SITE_ERROR)
        self.assertEqual(calls, [])
        self.assertEqual(len(report.stages), 1)
        self.assertEqual(report.stages[0].stage, "guard")
        self.assertIn("failed recently", report.stages[0].error)

    def test_hard_dead_link_stops_chain(self):
        # A stage that proves the LINK dead (404) stops the chain:
        # the later rungs never re-fetch the same dead URL.
        url = "https://soundcloud.com/dead-artist/dead-track-2"
        engine = MediaDownloader(_ctx(self._tmp), direct_retries=1)
        calls: list[str] = []

        def _dead_404(stage, report, *args, **kwargs):
            calls.append(stage)
            report.stages.append(StageLog(
                stage=stage, url=url, latency_s=0.0,
                error_class=FailureKind.SITE_ERROR.value,
                error="HTTP 404 Not Found — track removed",
                note="stage failed"))
            return False
        engine._attempt_stage = _dead_404  # noqa: SLF001
        report = engine.download(url, dest_dir=self._tmp)
        self.assertFalse(report.ok)
        self.assertEqual(calls, ["direct"])
        self.assertEqual(report.failure, FailureKind.SITE_ERROR)

    def test_exhaustion_records_url(self):
        # A chain that exhausts on remote-shaped failures
        # remembers the URL — the next caller refuses it fast.
        url = "https://soundcloud.com/dead-artist/dead-track-3"
        engine = MediaDownloader(_ctx(self._tmp), direct_retries=1)

        def _fail_500(stage, report, *args, **kwargs):
            report.stages.append(StageLog(
                stage=stage, url=url, latency_s=0.0,
                error_class=FailureKind.SITE_ERROR.value,
                error="HTTP 500 internal server error",
                note="stage failed"))
            return False
        engine._attempt_stage = _fail_500  # noqa: SLF001
        report = engine.download(url, dest_dir=self._tmp)
        self.assertFalse(report.ok)
        self.assertEqual(_df.dead_kind_for_url(url), "site_error")
        self.assertTrue(_df.is_url_dead(url))

    def test_local_write_failure_records_nothing(self):
        # A run whose stages all failed locally (disk full, no
        # directory) is evidence about this box, never the link —
        # the URL must not be condemned for 6 hours.
        url = "https://soundcloud.com/live-artist/live-track"
        engine = MediaDownloader(_ctx(self._tmp), direct_retries=1)

        def _fail_local(stage, report, *args, **kwargs):
            report.stages.append(StageLog(
                stage=stage, url=url, latency_s=0.0,
                error_class=FailureKind.LOCAL_WRITE_ERROR.value,
                error="no space left on device",
                note="stage failed"))
            return False
        engine._attempt_stage = _fail_local  # noqa: SLF001
        report = engine.download(url, dest_dir=self._tmp)
        self.assertFalse(report.ok)
        self.assertIsNone(_df.dead_kind_for_url(url))
        self.assertFalse(_df.is_url_dead(url))

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()


# ── (4) the YouTube wall ─────────────────────────────────────────────


class _FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL: captures targets, can fail."""

    seen_targets: list = []
    entries: list = []
    error: Exception | None = None

    def __init__(self, opts):
        self._opts = dict(opts or {})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, target, download=False):
        _FakeYDL.seen_targets.append(target)
        if _FakeYDL.error is not None:
            raise _FakeYDL.error
        return {"entries": list(_FakeYDL.entries)}


def _install_fake_ydl(monkeypatch, entries=None, error=None):
    fake_module = types.ModuleType("yt_dlp")
    fake_module.YoutubeDL = _FakeYDL
    _FakeYDL.seen_targets.clear()
    _FakeYDL.entries = list(entries or [])
    _FakeYDL.error = error
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
    return fake_module


class TestYoutubeBotWall(unittest.TestCase):
    def test_bot_block_recorded_and_gates_second_call(self):
        # The first leg hits the wall and records it; the second
        # leg never invokes yt-dlp again — one wall, one invocation.
        _install_fake_ydl(
            self._mp, error=Exception(
                "Sign in to confirm you're not a bot"))
        out = PlaybackEngine._youtube_search_many(
            "tell me one lie", limit=8)
        self.assertEqual(out, [])
        self.assertEqual(
            _df.last_search_failure("youtube"), "bot_block")
        self.assertEqual(len(_FakeYDL.seen_targets), 1)
        out2 = PlaybackEngine._youtube_search_many(
            "tell me one lie", limit=8)
        self.assertEqual(out2, [])
        self.assertEqual(len(_FakeYDL.seen_targets), 1)

    def test_search_id_records_bot_block_and_fails_fast(self):
        _install_fake_ydl(
            self._mp, error=Exception(
                "Sign in to confirm you're not a bot"))
        with self.assertRaises(ToolError):
            PlaybackEngine._youtube_search_id("tell me one lie")
        self.assertEqual(
            _df.last_search_failure("youtube"), "bot_block")
        self.assertEqual(len(_FakeYDL.seen_targets), 1)
        with self.assertRaises(ToolError) as caught:
            PlaybackEngine._youtube_search_id("tell me one lie")
        self.assertIn("bot-block", str(caught.exception))
        self.assertEqual(len(_FakeYDL.seen_targets), 1)

    def test_music_leg_shares_the_wall(self):
        # A bot check recorded on the YouTube leg gates the
        # YouTube Music leg too — same wall, same cookies, same IP.
        _df.record_search_failure("youtube", "bot_block")
        _install_fake_ydl(self._mp, entries=[{"id": "abc"}])
        out = PlaybackEngine._youtube_search_many(
            "tell me one lie", limit=8, prefix="ytmsearch")
        self.assertEqual(out, [])
        self.assertEqual(_FakeYDL.seen_targets, [])

    def setUp(self):
        self._mp = pytest.MonkeyPatch()

    def tearDown(self):
        self._mp.undo()
