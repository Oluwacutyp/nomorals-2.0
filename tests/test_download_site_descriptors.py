"""Downloads B4 pins — per-site descriptors, SoundCloud #1.

The download engine reads descriptors (``media.site_descriptors``):
a site is URL patterns + the chain of rungs that applies to it, in
order + the connector id that resolves its streams + the block
vocabulary it can produce + its auth requirement.  SoundCloud is
the first descriptor, migrated off the engine's hardcoded path
(the ``soundcloud.com`` host check in ``_stage_connector`` and the
ladder tuple in ``download()``).

Pins:
* the SoundCloud descriptor reproduces B1–B3 behaviour exactly —
  the B2 connector-rung tests re-run here against the descriptor
  path (progressive via kit, HLS via the direct backend, a
  classified block journaled with its class and the chain moving
  on, a non-SoundCloud URL never engaging the connector);
* a malformed descriptor fails at LOAD naming site + field, never
  mid-download;
* a second (toy) descriptor proves the engine is data-driven: a
  site's chain order, its connector, and its downloadable flag
  come from the descriptor, not the engine.

Everything is faked at the seams — no network, no browser engine.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.connectors.soundcloud import SoundCloudError
from nomorals.media.site_descriptors import (
    AUDIOMACK_DESCRIPTOR,
    BLOCK_CLASSES,
    BOOMPLAY_DESCRIPTOR,
    CHAIN_STAGES,
    MP3PM_DESCRIPTOR,
    NETNAIJA_DESCRIPTOR,
    SOUNDCLOUD_DESCRIPTOR,
    TIKTOK_DESCRIPTOR,
    YOUTUBE_DESCRIPTOR,
    SiteDescriptor,
    SiteDescriptorError,
    SiteDescriptorRegistry,
)

# ── shared fakes (same seams as tests/test_download_expansion.py) ───


class _FakeKit:
    def __init__(self, payload: bytes = b"connector-bytes"):
        self.payload = payload
        self.calls: list[str] = []

    def download(self, url, dest, progress=None):
        self.calls.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.payload)
        return SimpleNamespace(
            ok=True, path=str(dest), size_bytes=len(self.payload),
            error="", resumed=False, restarted=False)


def _no_descriptor_extractors():
    return SimpleNamespace(best_for=lambda url: None)


def _fail_direct(*a, **k):
    raise RuntimeError("connection reset by peer")


def _downloader(tmp, connector=None, *, site_descriptors=None,
                site_connectors=None):
    from nomorals.media.downloader import MediaDownloader

    return MediaDownloader(
        None, direct_retries=1, soundcloud_connector=connector,
        kit_downloader=_FakeKit(),
        extractors=_no_descriptor_extractors(),
        site_descriptors=site_descriptors,
        site_connectors=site_connectors)


# ── the SoundCloud descriptor itself ─────────────────────────────────


class TestSoundCloudDescriptor:
    def test_matches_soundcloud_urls_only(self):
        reg = SiteDescriptorRegistry([SOUNDCLOUD_DESCRIPTOR])
        assert reg.for_url("https://soundcloud.com/artist/track") \
            is SOUNDCLOUD_DESCRIPTOR
        assert reg.for_url(
            "https://www.soundcloud.com/artist/track") \
            is SOUNDCLOUD_DESCRIPTOR
        assert reg.for_url(
            "https://m.soundcloud.com/artist/track") \
            is SOUNDCLOUD_DESCRIPTOR
        assert reg.for_url("https://on.soundcloud.com/abc123") \
            is SOUNDCLOUD_DESCRIPTOR
        assert reg.for_url("https://example.com/watch?v=1") is None
        assert reg.for_url("https://notsoundcloud.com/x") is None
        assert reg.for_url("https://soundcloud.com.evil.test/x") is None

    def test_descriptor_carries_b1_b3_behaviour(self):
        desc = SOUNDCLOUD_DESCRIPTOR
        assert desc.site == "soundcloud"
        assert desc.chain == CHAIN_STAGES
        assert desc.chain == ("direct", "kit", "connector", "proxy",
                              "rendered", "browser")
        assert desc.connector == "soundcloud"
        assert tuple(desc.block_classes) == BLOCK_CLASSES
        assert desc.auth == "none"
        assert desc.auth_needed is False
        assert desc.downloadable is True

    def test_default_registry_is_the_soundcloud_descriptor(self):
        from nomorals.media.site_descriptors import (
            DEFAULT_SITE_DESCRIPTORS,
        )

        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://soundcloud.com/a/b") is SOUNDCLOUD_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.get("soundcloud") \
            is SOUNDCLOUD_DESCRIPTOR


# ── the YouTube descriptor (descriptor #2) ───────────────────────────


class TestYouTubeDescriptor:
    #: Real YouTube URL shapes, one per form the descriptor claims.
    YOUTUBE_URLS = (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/v/dQw4w9WgXcQ",
        "https://www.youtube.com/playlist?list=PL123456",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=abc123",
    )

    #: URLs the descriptor must NOT claim: non-YouTube hosts, look-
    #: alikes, and YouTube pages that are not a playable item (home,
    #: channel, search, feed) — those fall through to the generic
    #: chain instead of taking the YouTube chain.
    NOT_YOUTUBE_URLS = (
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://vimeo.com/123456",
        "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/",
        "https://www.youtube.com/@SomeChannel",
        "https://www.youtube.com/channel/UC123456",
        "https://www.youtube.com/results?search_query=cats",
        "https://www.youtube.com/feed/subscriptions",
        "https://youtu.be/",
    )

    def test_matches_youtube_url_shapes_only(self):
        reg = SiteDescriptorRegistry([YOUTUBE_DESCRIPTOR])
        for url in self.YOUTUBE_URLS:
            assert reg.for_url(url) is YOUTUBE_DESCRIPTOR, url
        for url in self.NOT_YOUTUBE_URLS:
            assert reg.for_url(url) is None, url

    def test_descriptor_carries_youtube_behaviour(self):
        desc = YOUTUBE_DESCRIPTOR
        assert desc.site == "youtube"
        # Canonical ladder minus the connector rung: there is no
        # YouTube connector, the kit rung (yt-dlp) resolves.
        assert desc.chain == ("direct", "kit", "proxy", "rendered",
                              "browser")
        assert "connector" not in desc.chain
        assert desc.connector == ""
        assert tuple(desc.block_classes) == (
            "region", "login", "private")
        # Public by default; the session shape is declared so a
        # login block can retry authenticated (B5).
        assert desc.auth == "none"
        assert desc.auth_needed is False
        assert desc.auth_domains == ("youtube.com",)
        assert desc.auth_cookie_names == ("SID", "SAPISID")
        assert desc.downloadable is True

    def test_default_registry_resolves_youtube_and_keeps_soundcloud(
            self):
        from nomorals.media.site_descriptors import (
            DEFAULT_SITE_DESCRIPTORS,
        )

        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ) is YOUTUBE_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://youtu.be/dQw4w9WgXcQ") is YOUTUBE_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.get("youtube") \
            is YOUTUBE_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://soundcloud.com/a/b") is SOUNDCLOUD_DESCRIPTOR

    def test_youtube_chain_skips_connector_and_fails_honestly(
            self, tmp_path):
        # The descriptor's chain governs the engine for a YouTube
        # URL: no rung in this hermetic rig can serve a watch page
        # — the kit rung only fetches direct-rule URLs (a watch
        # page is the ytdlp rule, not direct); there is no YouTube
        # connector; the proxy pool is empty; rendered and browser
        # are unwired — so the report fails honestly, and the
        # injected connector is never engaged.  In production the
        # rendered / browser rungs are what land a YouTube video
        # (driven-player capture / page extraction); a live landing
        # is NOT proven here — it needs network and a real video.
        from unittest.mock import patch

        connector = SimpleNamespace(
            stream_url=mock.Mock(
                side_effect=AssertionError("connector engaged")))
        registry = SiteDescriptorRegistry([YOUTUBE_DESCRIPTOR])
        dl = _downloader(tmp_path, connector,
                         site_descriptors=registry)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        # Kit silent (ytdlp rule, not direct), connector skipped.
        assert report.tried() == ["direct", "proxy", "rendered",
                                  "browser"]
        assert "connector" not in report.tried()
        connector.stream_url.assert_not_called()


# ── the TikTok descriptor (descriptor #3) ─────────────────────────


class TestTikTokDescriptor:
    #: Real TikTok URL shapes, one per form the descriptor claims.
    #: Verified against yt-dlp's own TikTok extractor grammar
    #: (yt_dlp/extractor/tiktok.py, master, fetched 2026-11-12):
    #: TikTokIE video (``@user/video/<id>``, ``share/video/<id>``,
    #: ``embed/<id>``, ``tiktokv.com``), TikTokVMIE short links
    #: (``vm|vt.tiktok.com`` + ``tiktok.com/t/``), TikTokLiveIE
    #: (``@user/live``, ``m.tiktok.com/share/live/<id>``) — plus
    #: ``/photo/<id>``, the photo-post form TikTok itself serves
    #: (real tiktok.com photo URLs), which yt-dlp does NOT cover:
    #: on a photo post it lands only the slideshow's background
    #: audio, never the images.
    TIKTOK_URLS = (
        "https://www.tiktok.com/@someuser/video/7170520270497680683",
        "https://www.tiktok.com/@someuser/video/7170520270497680683"
        "?is_from_webapp=1&sender_device=pc",
        "https://m.tiktok.com/@someuser/video/7170520270497680683",
        "https://www.tiktok.com/@someuser/photo/7604667694410894606",
        "https://www.tiktok.com/@someuser/live",
        "https://www.tiktok.com/share/video/7170520270497680683",
        "https://www.tiktok.com/embed/7170520270497680683",
        "https://www.tiktokv.com/embed/7170520270497680683",
        "https://www.tiktok.com/t/ZTRC5xgJp",
        "https://vm.tiktok.com/ZTR45GpSF/",
        "https://vt.tiktok.com/ZSe4FqkKd",
        "https://m.tiktok.com/share/live/7209423610325322522",
    )

    #: URLs the descriptor must NOT claim: non-TikTok hosts,
    #: look-alikes, bare short hosts, and TikTok pages that are
    #: not one playable item (home, profile, tag, music, feed) —
    #: those fall through to the generic chain instead of taking
    #: the TikTok chain.
    NOT_TIKTOK_URLS = (
        "https://example.com/@user/video/7170520270497680683",
        "https://www.instagram.com/reel/123456",
        "https://nottiktok.com/@user/video/7170520270497680683",
        "https://tiktok.com.evil.test/@user/video/7170520270497680683",
        "https://www.tiktok.com/",
        "https://www.tiktok.com/@someuser",
        "https://www.tiktok.com/@someuser/video/",
        "https://www.tiktok.com/@someuser/video",
        "https://www.tiktok.com/@someuser/livestream",
        "https://www.tiktok.com/tag/cats",
        "https://www.tiktok.com/music/original-sound-7170520270497680683",
        "https://www.tiktok.com/foryou",
        "https://www.tiktok.com/discover",
        "https://vm.tiktok.com/",
        "https://vt.tiktok.com/",
    )

    def test_matches_tiktok_url_shapes_only(self):
        reg = SiteDescriptorRegistry([TIKTOK_DESCRIPTOR])
        for url in self.TIKTOK_URLS:
            assert reg.for_url(url) is TIKTOK_DESCRIPTOR, url
        for url in self.NOT_TIKTOK_URLS:
            assert reg.for_url(url) is None, url

    def test_descriptor_carries_tiktok_behaviour(self):
        desc = TIKTOK_DESCRIPTOR
        assert desc.site == "tiktok"
        # Canonical ladder minus the connector rung: there is no
        # TikTok connector, the direct stage (yt-dlp) resolves.
        assert desc.chain == ("direct", "kit", "proxy", "rendered",
                              "browser")
        assert "connector" not in desc.chain
        assert desc.connector == ""
        assert tuple(desc.block_classes) == (
            "region", "login", "private")
        # Public by default; the session shape is declared so a
        # login block can retry authenticated (B5) — sid_tt is the
        # one cookie yt-dlp's TikTok extractor reads and forwards
        # to the API host; sessionid / msToken are not consumed by
        # that extractor and are not named.
        assert desc.auth == "none"
        assert desc.auth_needed is False
        assert desc.auth_domains == ("tiktok.com",)
        assert desc.auth_cookie_names == ("sid_tt",)
        assert desc.downloadable is True

    def test_default_registry_resolves_tiktok_and_keeps_the_others(
            self):
        from nomorals.media.site_descriptors import (
            DEFAULT_SITE_DESCRIPTORS,
        )

        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.tiktok.com/@someuser/video/7170520270497680683"
            ) is TIKTOK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://vm.tiktok.com/ZTR45GpSF/") is TIKTOK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.tiktok.com/@someuser/photo/7604667694410894606"
            ) is TIKTOK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.get("tiktok") \
            is TIKTOK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ) is YOUTUBE_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://soundcloud.com/a/b") is SOUNDCLOUD_DESCRIPTOR

    def test_tiktok_chain_skips_connector_and_fails_honestly(
            self, tmp_path):
        # The descriptor's chain governs the engine for a TikTok
        # URL: no rung in this hermetic rig can serve a video page
        # — the kit rung only fetches direct-rule URLs (a video
        # page is the ytdlp rule, not direct); there is no TikTok
        # connector; the proxy pool is empty; rendered and browser
        # are unwired — so the report fails honestly, and the
        # injected connector is never engaged.  In production the
        # direct stage (yt-dlp) lands a TikTok video; a live
        # landing is NOT proven here — it needs network and a real
        # video.  A photo post lands audio-only via yt-dlp, never
        # its images — that limit is named on the descriptor, not
        # hidden by this test.
        from unittest.mock import patch

        connector = SimpleNamespace(
            stream_url=mock.Mock(
                side_effect=AssertionError("connector engaged")))
        registry = SiteDescriptorRegistry([TIKTOK_DESCRIPTOR])
        dl = _downloader(tmp_path, connector,
                         site_descriptors=registry)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://www.tiktok.com/@someuser/video/"
                "7170520270497680683",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        # Kit silent (ytdlp rule, not direct), connector skipped.
        assert report.tried() == ["direct", "proxy", "rendered",
                                  "browser"]
        assert "connector" not in report.tried()
        connector.stream_url.assert_not_called()


# ── the mp3.pm descriptor (descriptor #4) ─────────────────────────


class TestMp3pmDescriptor:
    #: Real mp3.pm URL shapes, one per form the descriptor claims.
    #: Every shape verified live 2026-11-12: the track page
    #: (``/song/170547096/T.I._-_King_Back/``), the search page
    #: (``/search/s/f/king+back+t+i/``), the genre chart
    #: (``/top/rock/``), and the per-artist subdomains
    #: (``9545933-dua-lipa.mp3.pm``, ``131-t-i.mp3.pm``,
    #: ``60437-c.mp3.pm``) — all served by the live site.
    MP3PM_URLS = (
        "https://mp3.pm/song/170547096/T.I._-_King_Back/",
        "https://mp3.pm/song/170547096/T.I._-_King_Back",
        "http://mp3.pm/song/12345/Artist_-_Title/",
        "https://www.mp3.pm/song/12345/A_-_B/",
        "https://mp3.pm/search/s/f/king+back+t+i/",
        "https://mp3.pm/search/s/f/drake",
        "https://mp3.pm/top/rock/",
        "https://mp3.pm/top/rap-hip-hop/",
        "https://9545933-dua-lipa.mp3.pm/",
        "https://131-t-i.mp3.pm/",
        "https://60437-c.mp3.pm/",
        "http://9545933-dua-lipa.mp3.pm/",
    )

    #: URLs the descriptor must NOT claim: non-mp3.pm hosts,
    #: look-alikes, the CDN-ish host, and mp3.pm pages that are
    #: not one playable item (home, bare search / genre / song,
    #: and online radio — streaming-only, no download option) —
    #: those fall through to the generic chain instead of taking
    #: the mp3.pm chain.
    NOT_MP3PM_URLS = (
        "https://example.com/song/170547096/T.I._-_King_Back/",
        "https://notmp3.pm/song/170547096/X/",
        "https://mp3.pm.evil.test/song/170547096/X/",
        "https://cs1.mp3.pm/",
        "https://mp3.pm/",
        "https://www.mp3.pm/",
        "https://mp3.pm/radio/pop/",
        "https://mp3.pm/top/",
        "https://mp3.pm/search/",
        "https://mp3.pm/search/s/f/",
        "https://mp3.pm/song/",
        "https://mp3.pm/song/abc/",
    )

    def test_matches_mp3pm_url_shapes_only(self):
        reg = SiteDescriptorRegistry([MP3PM_DESCRIPTOR])
        for url in self.MP3PM_URLS:
            assert reg.for_url(url) is MP3PM_DESCRIPTOR, url
        for url in self.NOT_MP3PM_URLS:
            assert reg.for_url(url) is None, url

    def test_descriptor_carries_mp3pm_behaviour(self):
        desc = MP3PM_DESCRIPTOR
        assert desc.site == "mp3pm"
        # Canonical ladder minus the connector rung: there is no
        # mp3.pm connector, and yt-dlp has no mp3.pm extractor
        # (supportedsites.md, master, 2026-11-12) — the kit rung
        # only fetches direct-rule file URLs and stays silent on
        # track pages; rendered / browser land a track.
        assert desc.chain == ("direct", "kit", "proxy", "rendered",
                              "browser")
        assert "connector" not in desc.chain
        assert desc.connector == ""
        # Only "private" is evidenced: a removed track.  No login
        # wall exists on this site at all — no region / login /
        # preview / drm classes.
        assert tuple(desc.block_classes) == ("private",)
        # Fully public AND session-less: no accounts exist, so no
        # session shape is declared — empty on purpose, not missing.
        assert desc.auth == "none"
        assert desc.auth_needed is False
        assert desc.auth_domains == ()
        assert desc.auth_cookie_names == ()
        assert desc.downloadable is True

    def test_default_registry_resolves_mp3pm_and_keeps_the_others(
            self):
        from nomorals.media.site_descriptors import (
            DEFAULT_SITE_DESCRIPTORS,
        )

        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://mp3.pm/song/170547096/T.I._-_King_Back/"
            ) is MP3PM_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://9545933-dua-lipa.mp3.pm/") is MP3PM_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://mp3.pm/search/s/f/king+back+t+i/"
            ) is MP3PM_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.get("mp3pm") \
            is MP3PM_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.tiktok.com/@someuser/video/"
            "7170520270497680683") is TIKTOK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ) is YOUTUBE_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://soundcloud.com/a/b") is SOUNDCLOUD_DESCRIPTOR

    def test_mp3pm_chain_skips_connector_and_fails_honestly(
            self, tmp_path):
        # The descriptor's chain governs the engine for an mp3.pm
        # URL: no rung in this hermetic rig can serve a track page
        # — the kit rung only fetches direct-rule URLs (a track
        # page is a page, not a file); there is no mp3.pm
        # connector; the proxy pool is empty; rendered and browser
        # are unwired — so the report fails honestly, and the
        # injected connector is never engaged.  In production the
        # rendered / browser rungs are what land a track (page
        # extraction); a live landing is NOT proven here — it needs
        # network and a real track.
        from unittest.mock import patch

        connector = SimpleNamespace(
            stream_url=mock.Mock(
                side_effect=AssertionError("connector engaged")))
        registry = SiteDescriptorRegistry([MP3PM_DESCRIPTOR])
        dl = _downloader(tmp_path, connector,
                         site_descriptors=registry)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://mp3.pm/song/170547096/T.I._-_King_Back/",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        # Kit silent (page, not a direct-rule file), connector
        # skipped.
        assert report.tried() == ["direct", "proxy", "rendered",
                                  "browser"]
        assert "connector" not in report.tried()
        connector.stream_url.assert_not_called()


class TestAudiomackDescriptor:
    #: Real Audiomack URL shapes, one per form the descriptor
    #: claims.  Every shape verified live 2026-11-12: the track
    #: pages (``/jstontrapp_/song/kwabena-adu``,
    #: ``/41/song/mvp-audiomack-fine-tuned-version-8135016``,
    #: ``/King-Maj/song/nack-am-prod-by-hitz-duktor``,
    #: ``/early-morning-jazz-playlist/song/cocktail-music-bar``),
    #: the bare ``/song/<artist>/<slug>`` form yt-dlp's own
    #: tests use (``/song/roosh-williams/extraordinary``), the
    #: album pages (``/ajujusamuel/album/beyond-limitations``,
    #: ``/ricoace/album/blackjack-8414236``,
    #: ``/album/flytunezcom/tha-tour-part-2-mixtape``), the
    #: playlist pages (``/future/playlist/future``,
    #: ``/joevango/playlist/lil-skies-essentials``,
    #: ``/gucci-mane/playlist/guccimane``), and the embed
    #: player (``/embed/eugeneportman/playlist/playlist``,
    #: ``/embed/j-balvin/playlist/j-balvin-essentials``) — all
    #: served by the live site (HTTP 200).
    AUDIOMACK_URLS = (
        "https://audiomack.com/jstontrapp_/song/kwabena-adu",
        "https://audiomack.com/41/song/"
        "mvp-audiomack-fine-tuned-version-8135016",
        "https://audiomack.com/King-Maj/song/"
        "nack-am-prod-by-hitz-duktor",
        "https://audiomack.com/early-morning-jazz-playlist/song/"
        "cocktail-music-bar",
        "http://www.audiomack.com/song/roosh-williams/extraordinary",
        "https://audiomack.com/ajujusamuel/album/beyond-limitations",
        "https://audiomack.com/ricoace/album/blackjack-8414236",
        "http://www.audiomack.com/album/flytunezcom/"
        "tha-tour-part-2-mixtape",
        "https://audiomack.com/future/playlist/future",
        "https://audiomack.com/joevango/playlist/"
        "lil-skies-essentials",
        "https://audiomack.com/gucci-mane/playlist/guccimane",
        "https://audiomack.com/embed/eugeneportman/playlist/playlist",
        "https://audiomack.com/embed/j-balvin/playlist/"
        "j-balvin-essentials",
        "https://audiomack.com/acedadon-savage/album/my-own-fan-4-2"
        "?share-user-id=125739655",
    )

    #: URLs the descriptor must NOT claim: non-Audiomack hosts,
    #: look-alikes, and Audiomack pages that are not one
    #: playable item (home, an artist page, search, and bare
    #: song / album / playlist paths) — those fall through to
    #: the generic chain instead of taking the Audiomack chain.
    NOT_AUDIOMACK_URLS = (
        "https://example.com/future/song/x",
        "https://notaudiomack.com/future/song/x",
        "https://audiomack.com.evil.test/future/song/x",
        "https://audiomack.com/",
        "https://www.audiomack.com/",
        "https://audiomack.com/future",
        "https://audiomack.com/search?q=drake",
        "https://audiomack.com/future/song/",
        "https://audiomack.com/song/",
        "https://audiomack.com/future/album/",
        "https://audiomack.com/future/playlist/",
    )

    def test_matches_audiomack_url_shapes_only(self):
        reg = SiteDescriptorRegistry([AUDIOMACK_DESCRIPTOR])
        for url in self.AUDIOMACK_URLS:
            assert reg.for_url(url) is AUDIOMACK_DESCRIPTOR, url
        for url in self.NOT_AUDIOMACK_URLS:
            assert reg.for_url(url) is None, url

    def test_descriptor_carries_audiomack_behaviour(self):
        desc = AUDIOMACK_DESCRIPTOR
        assert desc.site == "audiomack"
        # Canonical ladder minus the connector rung: there is no
        # Audiomack connector, and the kit rung only fetches
        # direct-rule file URLs and stays silent on pages — in
        # production the direct stage (yt-dlp's audiomack /
        # audiomack:album extractors, no auth) lands a track or
        # album.
        assert desc.chain == ("direct", "kit", "proxy", "rendered",
                              "browser")
        assert "connector" not in desc.chain
        assert desc.connector == ""
        # Only "private" is evidenced: a removed track.  No login
        # wall exists on public playback and yt-dlp's extractor
        # reads no cookies at all — so no login / region / preview
        # / drm classes.  Audiomack+ premium-only tracks are a
        # paywall with no block class and fail honestly
        # unclassified, never as a named block.
        assert tuple(desc.block_classes) == ("private",)
        # Fully public AND session-less: no session shape is
        # declared — empty on purpose, not missing.  A login block
        # can never be classified for this site.
        assert desc.auth == "none"
        assert desc.auth_needed is False
        assert desc.auth_domains == ()
        assert desc.auth_cookie_names == ()
        assert desc.downloadable is True

    def test_default_registry_resolves_audiomack_and_keeps_the_others(
            self):
        from nomorals.media.site_descriptors import (
            DEFAULT_SITE_DESCRIPTORS,
        )

        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://audiomack.com/jstontrapp_/song/kwabena-adu"
            ) is AUDIOMACK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://audiomack.com/ajujusamuel/album/"
            "beyond-limitations") is AUDIOMACK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://audiomack.com/future/playlist/future"
            ) is AUDIOMACK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.get("audiomack") \
            is AUDIOMACK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://mp3.pm/song/170547096/T.I._-_King_Back/"
            ) is MP3PM_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.tiktok.com/@someuser/video/"
            "7170520270497680683") is TIKTOK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ) is YOUTUBE_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://soundcloud.com/a/b") is SOUNDCLOUD_DESCRIPTOR

    def test_audiomack_chain_skips_connector_and_fails_honestly(
            self, tmp_path):
        # The descriptor's chain governs the engine for an
        # Audiomack URL: no rung in this hermetic rig can serve a
        # track page — the kit rung only fetches direct-rule URLs
        # (a track page is a page, not a file); there is no
        # Audiomack connector; the proxy pool is empty; rendered
        # and browser are unwired — so the report fails honestly,
        # and the injected connector is never engaged.  In
        # production the direct stage (yt-dlp's audiomack
        # extractor, no auth) is what lands a track; a live
        # landing is NOT proven here — it needs network and a
        # real track.
        from unittest.mock import patch

        connector = SimpleNamespace(
            stream_url=mock.Mock(
                side_effect=AssertionError("connector engaged")))
        registry = SiteDescriptorRegistry([AUDIOMACK_DESCRIPTOR])
        dl = _downloader(tmp_path, connector,
                         site_descriptors=registry)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://audiomack.com/jstontrapp_/song/kwabena-adu",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        # Kit silent (page, not a direct-rule file), connector
        # skipped.
        assert report.tried() == ["direct", "proxy", "rendered",
                                  "browser"]
        assert "connector" not in report.tried()
        connector.stream_url.assert_not_called()


# ── the NetNaija descriptor (descriptor #6) ─────────────────────


class TestNetnaijaDescriptor:
    #: Real NetNaija URL shapes, one per form the descriptor
    #: claims.  Provenance is in the descriptor comment — the
    #: live site could NOT be fetched from this box (geo-block:
    #: netnaija.com answers an empty reply, thenetnaija.com
    #: answers HTTP 521 via Cloudflare, 2026-11-12), so the
    #: shapes come from the in-repo source and its tests, not a
    #: live fetch: post pages ``/music/burna-boy-last-last/`` and
    #: ``/music/ayra-starr-rush/`` (``tests/test_audio_sources.py``
    #: fixtures), the ``/?s=`` search (``NetNaijaSource.search_url``),
    #: the ``/dl/<id>/`` hop and the ``dl.`` / ``files.`` direct
    #: MP3s (source-test fixtures).  The ``thenetnaija.com``
    #: post shape is the same WordPress permalink on the site's
    #: current domain, whose ``/music/`` category is verified
    #: live (cached crawl 2025-06-01; Feedspot music feed).
    NETNAIJA_URLS = (
        "https://netnaija.com/music/burna-boy-last-last/",
        "https://netnaija.com/music/ayra-starr-rush/",
        "http://netnaija.com/music/x/",
        "https://www.netnaija.com/music/burna-boy-last-last/",
        "https://www.thenetnaija.com/music/burna-boy-last-last/",
        "https://thenetnaija.com/music/ayra-starr-rush/",
        "https://netnaija.com/?s=davido",
        "https://www.thenetnaija.com/?s=burna+boy",
        "https://netnaija.com/dl/abc123/",
        "https://dl.netnaija.com/music/last-last.mp3",
        "https://files.netnaija.com/x/last-last.mp3",
        "http://files.netnaija.com/x/last-last.mp3",
    )

    #: URLs the descriptor must NOT claim: non-NetNaija hosts,
    #: look-alikes, the film host ``thenetnaija.net`` (the film
    #: chain's seed — not the music descriptor's), and NetNaija
    #: pages that are not one playable item (home, the ``/music/``
    #: category listing, a bare search, a bare ``/dl/`` hop, and
    #: movie posts — the film chain's) — those fall through to
    #: the generic chain instead of taking the NetNaija chain.
    NOT_NETNAIJA_URLS = (
        "https://example.com/music/burna-boy-last-last/",
        "https://notnetnaija.com/music/x/",
        "https://netnaija.com.evil.test/music/x/",
        "https://netnaija.com/",
        "https://www.netnaija.com/",
        "https://netnaija.com/music/",
        "https://thenetnaija.com/music/",
        "https://netnaija.com/?s=",
        "https://netnaija.com/dl/",
        "https://netnaija.com/movies/some-film/",
        "https://thenetnaija.net/?s=davido",
        "https://thenetnaija.net/music/x/",
        # A non-MP3 file off the music path: neither the post
        # pattern (no /music/ path) nor the file pattern (no
        # .mp3) claims it.
        "https://dl.netnaija.com/other/last-last.wav",
    )

    def test_matches_netnaija_url_shapes_only(self):
        reg = SiteDescriptorRegistry([NETNAIJA_DESCRIPTOR])
        for url in self.NETNAIJA_URLS:
            assert reg.for_url(url) is NETNAIJA_DESCRIPTOR, url
        for url in self.NOT_NETNAIJA_URLS:
            assert reg.for_url(url) is None, url

    def test_descriptor_carries_netnaija_behaviour(self):
        desc = NETNAIJA_DESCRIPTOR
        assert desc.site == "netnaija"
        # Canonical ladder minus the connector rung: there is no
        # NetNaija connector, and yt-dlp has no NetNaija
        # extractor (supportedsites.md, master, 2026-11-12) — the
        # kit rung only fetches direct-rule file URLs (the dl. /
        # files. MP3s) and stays silent on post pages; rendered /
        # browser land a post in production.
        assert desc.chain == ("direct", "kit", "proxy", "rendered",
                              "browser")
        assert "connector" not in desc.chain
        assert desc.connector == ""
        # "region" is evidenced: the whole site geo-blocks
        # non-NG networks (empty reply / HTTP 521 from this box;
        # the film host was 403 bot-blocked in the 2026-10-18
        # verification).  "private" is a removed post.  No login
        # wall exists on this public WordPress blog — so no login
        # / preview / drm classes.
        assert tuple(desc.block_classes) == ("region", "private")
        # Fully public AND session-less: no accounts exist, so no
        # session shape is declared — empty on purpose, not missing.
        assert desc.auth == "none"
        assert desc.auth_needed is False
        assert desc.auth_domains == ()
        assert desc.auth_cookie_names == ()
        assert desc.downloadable is True

    def test_default_registry_resolves_netnaija_and_keeps_the_others(
            self):
        from nomorals.media.site_descriptors import (
            DEFAULT_SITE_DESCRIPTORS,
        )

        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://netnaija.com/music/burna-boy-last-last/"
            ) is NETNAIJA_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://thenetnaija.com/music/ayra-starr-rush/"
            ) is NETNAIJA_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://netnaija.com/?s=davido"
            ) is NETNAIJA_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://dl.netnaija.com/music/last-last.mp3"
            ) is NETNAIJA_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.get("netnaija") \
            is NETNAIJA_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://audiomack.com/jstontrapp_/song/kwabena-adu"
            ) is AUDIOMACK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://mp3.pm/song/170547096/T.I._-_King_Back/"
            ) is MP3PM_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.tiktok.com/@someuser/video/"
            "7170520270497680683") is TIKTOK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ) is YOUTUBE_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://soundcloud.com/a/b") is SOUNDCLOUD_DESCRIPTOR

    def test_netnaija_chain_skips_connector_and_fails_honestly(
            self, tmp_path):
        # The descriptor's chain governs the engine for a
        # NetNaija URL: no rung in this hermetic rig can serve a
        # post page — the kit rung only fetches direct-rule URLs
        # (a post page is a page, not a file); there is no
        # NetNaija connector; the proxy pool is empty; rendered
        # and browser are unwired — so the report fails honestly,
        # and the injected connector is never engaged.  In
        # production the rendered / browser rungs are what land a
        # post (page extraction); a live landing is NOT proven
        # here — it needs a Nigerian network and a real track,
        # and the site is unreachable from this box.
        from unittest.mock import patch

        connector = SimpleNamespace(
            stream_url=mock.Mock(
                side_effect=AssertionError("connector engaged")))
        registry = SiteDescriptorRegistry([NETNAIJA_DESCRIPTOR])
        dl = _downloader(tmp_path, connector,
                         site_descriptors=registry)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://netnaija.com/music/burna-boy-last-last/",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        # Kit silent (page, not a direct-rule file), connector
        # skipped.
        assert report.tried() == ["direct", "proxy", "rendered",
                                  "browser"]
        assert "connector" not in report.tried()
        connector.stream_url.assert_not_called()


class TestBoomplayDescriptor:
    #: Real Boomplay URL shapes, one per form the descriptor
    #: claims.  Provenance is in the descriptor comment — all
    #: verified live 2026-11-12: track ``/songs/132072…`` (a
    #: live-linked song URL) and the ``<id>_<slug>`` form from
    #: the in-repo source's legacy path; album ``/albums/
    #: 93328816`` and ``/albums/56593624``; artist
    #: ``/artists/55663392`` and ``/artists/42423194?from=search``.
    BOOMPLAY_URLS = (
        "https://www.boomplay.com/songs/132072",
        "https://www.boomplay.com/songs/12345678_last-last/",
        "http://boomplay.com/songs/12345678",
        "https://boomplay.com/songs/12345678",
        "https://www.boomplay.com/albums/93328816",
        "https://www.boomplay.com/albums/56593624",
        "https://boomplay.com/albums/55665594",
        "https://www.boomplay.com/artists/55663392",
        "https://www.boomplay.com/artists/42423194?from=search",
        "https://www.boomplay.com/albums/42424763?from=search",
    )

    #: URLs the descriptor must NOT claim: non-Boomplay hosts,
    #: look-alikes, and Boomplay pages that are not one content
    #: item (home, search — ``/search/music/`` is a listing and
    #: the legacy ``/search?q=`` answers a login page to an
    #: anonymous fetch — bare ``/songs/`` / ``/albums/`` /
    #: ``/artists/`` with no id, playlist pages whose
    #: ``/playlists/<id>`` shape was not verified in this lane,
    #: and chart listings).
    NOT_BOOMPLAY_URLS = (
        "https://example.com/songs/132072",
        "https://notboomplay.com/songs/123/",
        "https://boomplay.com.evil.test/songs/123/",
        "https://www.boomplay.com/",
        "https://boomplay.com/",
        "https://www.boomplay.com/search/music/burna%20boy",
        "https://www.boomplay.com/search?q=burna+boy",
        "https://www.boomplay.com/songs/",
        "https://www.boomplay.com/albums/",
        "https://www.boomplay.com/artists/",
        "https://www.boomplay.com/playlists/12345678",
        "https://www.boomplay.com/charts/top-songs",
    )

    def test_matches_boomplay_url_shapes_only(self):
        reg = SiteDescriptorRegistry([BOOMPLAY_DESCRIPTOR])
        for url in self.BOOMPLAY_URLS:
            assert reg.for_url(url) is BOOMPLAY_DESCRIPTOR, url
        for url in self.NOT_BOOMPLAY_URLS:
            assert reg.for_url(url) is None, url

    def test_descriptor_carries_boomplay_behaviour(self):
        desc = BOOMPLAY_DESCRIPTOR
        assert desc.site == "boomplay"
        # Canonical ladder minus the connector rung: there is no
        # Boomplay connector and yt-dlp has no Boomplay extractor
        # (supportedsites.md, master, 2026-11-12) — declared for
        # shape-completeness, but the chain NEVER runs: the
        # site-descriptor refusal gate fires first.
        assert desc.chain == ("direct", "kit", "proxy", "rendered",
                              "browser")
        assert "connector" not in desc.chain
        assert desc.connector == ""
        # "private" is a removed track.  No region (no geo wall
        # seen — homepage and search served from this non-NG
        # box), no login (no login wall on the claimed pages),
        # no preview / drm (streams are protected wholesale, not
        # preview-clipped or DRM-classified).
        assert tuple(desc.block_classes) == ("private",)
        # Fully public AND session-less: public pages serve
        # without a session, and the SPA search's JSESSIONID /
        # imei cookies are anonymous, not a signed-in session —
        # so no session shape is declared, empty on purpose.
        assert desc.auth == "none"
        assert desc.auth_needed is False
        assert desc.auth_domains == ()
        assert desc.auth_cookie_names == ()
        # THE point of this descriptor: Boomplay serves no
        # downloads at all.
        assert desc.downloadable is False
        assert "serves no downloads" in desc.notes

    def test_default_registry_resolves_boomplay_and_keeps_the_others(
            self):
        from nomorals.media.site_descriptors import (
            DEFAULT_SITE_DESCRIPTORS,
        )

        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.boomplay.com/songs/132072"
            ) is BOOMPLAY_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.boomplay.com/albums/93328816"
            ) is BOOMPLAY_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.boomplay.com/artists/55663392"
            ) is BOOMPLAY_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.get("boomplay") \
            is BOOMPLAY_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://netnaija.com/music/burna-boy-last-last/"
            ) is NETNAIJA_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://audiomack.com/jstontrapp_/song/kwabena-adu"
            ) is AUDIOMACK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://mp3.pm/song/170547096/T.I._-_King_Back/"
            ) is MP3PM_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.tiktok.com/@someuser/video/"
            "7170520270497680683") is TIKTOK_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ) is YOUTUBE_DESCRIPTOR
        assert DEFAULT_SITE_DESCRIPTORS.for_url(
            "https://soundcloud.com/a/b") is SOUNDCLOUD_DESCRIPTOR

    def test_boomplay_url_refused_honestly_not_fallen_through(
            self, tmp_path):
        # The FIRST downloadable=False descriptor: a Boomplay URL
        # is refused by the site-descriptor gate with Boomplay's
        # own reason — the chain never runs, the generic fallback
        # never gets a chance, and the report names the site and
        # the app instead of a bare failure.  The default registry
        # is used (no injection) so this proves the shipped
        # wiring, not a test-only registry.  A live landing is NOT
        # proven here — there is nothing to land; the refusal is
        # the behaviour.
        from unittest.mock import patch

        dl = _downloader(tmp_path)
        with patch("nomorals.tools.media.download",
                   side_effect=AssertionError("chain ran")):
            report = dl.download(
                "https://www.boomplay.com/songs/132072",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        assert report.extractor == "boomplay"
        assert "boomplay" in report.likely_cause
        assert "does not serve downloads" in report.likely_cause
        assert "Boomplay app" in report.likely_cause
        assert report.tried() == ["extractor"]
        assert "downloadable source" in report.hint


# ── validation: malformed fails at LOAD, naming site + field ─────────


class TestDescriptorValidation:
    def _bad(self, **overrides):
        base = {
            "site": "soundcloud",
            "url_patterns": (r"soundcloud\.com",),
            "chain": CHAIN_STAGES,
            "connector": "soundcloud",
            "block_classes": BLOCK_CLASSES,
            "auth": "none",
        }
        base.update(overrides)
        return base

    def test_bad_regex_names_site_and_field(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor(**self._bad(
                url_patterns=(r"soundcloud\.com(",)))
        assert "soundcloud" in str(ei.value)
        assert "url_patterns" in str(ei.value)

    def test_empty_patterns_names_field(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor(**self._bad(url_patterns=()))
        assert "soundcloud" in str(ei.value)
        assert "url_patterns" in str(ei.value)

    def test_unknown_rung_names_field(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor(**self._bad(chain=("direct", "teleport")))
        assert "soundcloud" in str(ei.value)
        assert "chain" in str(ei.value)
        assert "teleport" in str(ei.value)

    def test_empty_chain_names_field(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor(**self._bad(chain=()))
        assert "chain" in str(ei.value)

    def test_connector_rung_without_connector_id_names_field(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor(**self._bad(connector=""))
        assert "soundcloud" in str(ei.value)
        assert "connector" in str(ei.value)

    def test_unknown_block_class_names_field(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor(**self._bad(
                block_classes=("preview", "geo-blocked")))
        assert "block_classes" in str(ei.value)
        assert "geo-blocked" in str(ei.value)

    def test_unknown_auth_names_field(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor(**self._bad(auth="oauth2"))
        assert "auth" in str(ei.value)

    def test_empty_site_names_field(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor(**self._bad(site=""))
        assert "site" in str(ei.value)

    def test_registry_validates_at_load_not_at_download(self):
        good = SiteDescriptor(**self._bad())
        # Corrupt after construction (frozen bypass): the registry
        # must still refuse it at LOAD.
        object.__setattr__(good, "chain", ("direct", "teleport"))
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptorRegistry([good])
        assert "chain" in str(ei.value)

    def test_register_validates_too(self):
        reg = SiteDescriptorRegistry()
        good = SiteDescriptor(**self._bad())
        object.__setattr__(good, "auth", "oauth2")
        with pytest.raises(SiteDescriptorError) as ei:
            reg.register(good)
        assert "auth" in str(ei.value)

    def test_from_dict_round_trip(self):
        desc = SiteDescriptor.from_dict({
            "site": "tunes", "url_patterns": [r"tunes\.test/"],
            "chain": ["connector"], "connector": "tunes",
            "block_classes": ["preview"], "auth": "account",
            "auth_domains": ["tunes.test"],
            "auth_cookie_names": ["sess"],
            "downloadable": True, "notes": "toy",
        })
        assert desc.site == "tunes"
        assert desc.auth_needed is True
        assert desc.auth_domains == ("tunes.test",)
        assert desc.auth_cookie_names == ("sess",)
        assert SiteDescriptor.from_dict(desc.to_dict()) == desc
        assert desc.to_dict()["site"] == "tunes"
        assert SiteDescriptor.from_dict(desc.to_dict()) == desc

    def test_from_dict_rejects_unknown_key_and_missing_required(self):
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor.from_dict({
                "site": "tunes", "url_patterns": [r"tunes\.test/"],
                "teleport": True})
        assert "teleport" in str(ei.value)
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor.from_dict({"url_patterns": [r"x"]})
        assert "site" in str(ei.value)
        with pytest.raises(SiteDescriptorError) as ei:
            SiteDescriptor.from_dict({"site": "tunes"})
        assert "url_patterns" in str(ei.value)


# ── the engine reads descriptors: SoundCloud behaviour identical ─────


class TestDescriptorDrivenSoundCloud:
    def test_stream_url_is_a_rung_after_direct(self, tmp_path):
        from unittest.mock import patch

        connector = SimpleNamespace(stream_url=lambda url: {
            "url": "https://cf-media.sndcdn.com/prog.mp3",
            "protocol": "progressive", "mime_type": "audio/mpeg",
            "track": {"title": "Midnight Drive",
                      "permalink_url":
                          "https://soundcloud.com/artist/midnight-drive"}})
        dl = _downloader(tmp_path, connector)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct):
            report = dl.download(
                "https://soundcloud.com/artist/midnight-drive",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
        assert report.extractor == "soundcloud-connector"
        assert report.title == "Midnight Drive"
        assert dl._kit().calls == ["https://cf-media.sndcdn.com/prog.mp3"]
        assert report.tried() == ["direct", "connector"]
        assert Path(report.path).read_bytes() == b"connector-bytes"

    def test_hls_stream_via_direct_backend(self, tmp_path):
        from unittest.mock import patch

        playlist = "https://cf-hls.sndcdn.com/pl.m3u8"
        connector = SimpleNamespace(stream_url=lambda url: {
            "url": playlist, "protocol": "hls",
            "mime_type": "application/x-mpegURL",
            "track": {"title": "HLS Track",
                      "permalink_url":
                          "https://soundcloud.com/artist/hls"}})

        def fake_direct(url, dest_dir, **kw):
            if url == playlist:
                target = Path(dest_dir) / "hls.mp3"
                target.write_bytes(b"hls-bytes")
                return {"path": str(target), "bytes": len(b"hls-bytes"),
                        "title": "HLS Track"}
            raise RuntimeError("connection reset by peer")

        dl = _downloader(tmp_path, connector)
        with patch("nomorals.tools.media.download",
                   side_effect=fake_direct):
            report = dl.download(
                "https://soundcloud.com/artist/hls",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
        assert report.extractor == "soundcloud-connector"
        assert report.tried() == ["direct", "connector"]
        assert Path(report.path).read_bytes() == b"hls-bytes"

    def test_blocked_connector_journals_class_and_falls_through(
            self, tmp_path):
        from unittest.mock import patch

        from nomorals.media.downloader import StageLog

        def blocked(url):
            raise SoundCloudError(
                "no playable stream: SoundCloud blocks this track "
                "in this region", reason="region")

        dl = _downloader(tmp_path, SimpleNamespace(stream_url=blocked))
        order = []

        def browser(report, *a, **k):
            order.append("browser")
            report.stages.append(StageLog(stage="browser",
                                          note="mocked"))
            return False

        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct), patch.object(
                       dl, "_stage_browser", side_effect=browser):
            report = dl.download(
                "https://soundcloud.com/artist/gated",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        conn_logs = [s for s in report.stages if s.stage == "connector"]
        assert conn_logs and "region" in conn_logs[0].note
        assert order == ["browser"]
        assert report.tried()[-1] == "browser"

    def test_non_soundcloud_url_never_engages_connector(self, tmp_path):
        from unittest.mock import patch

        connector = SimpleNamespace(
            stream_url=mock.Mock(side_effect=AssertionError("engaged")))
        dl = _downloader(tmp_path, connector)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct), patch.object(
                       dl, "_stage_browser", return_value=False):
            report = dl.download("https://example.com/watch?v=1",
                                 dest_dir=tmp_path)
        assert report.ok is False
        assert "connector" not in report.tried()
        connector.stream_url.assert_not_called()

    def test_no_factory_records_absence_and_moves_on(self, tmp_path):
        from unittest.mock import patch

        from nomorals.media.downloader import MediaDownloader

        dl = MediaDownloader(
            None, direct_retries=1,
            kit_downloader=_FakeKit(),
            extractors=_no_descriptor_extractors())
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct), patch.object(
                       dl, "_stage_browser", return_value=False):
            report = dl.download(
                "https://soundcloud.com/artist/track",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        conn_logs = [s for s in report.stages if s.stage == "connector"]
        assert conn_logs and "unavailable" in conn_logs[0].note


# ── a second descriptor proves the engine is data-driven ─────────────


def _toy_descriptor(**overrides: Any) -> SiteDescriptor:
    base: dict[str, Any] = {
        "site": "exampletunes",
        "url_patterns": (r"https?://([a-z0-9-]+\.)*exampletunes\.test/",),
        "chain": ("connector",),
        "connector": "exampletunes",
        "block_classes": ("preview",),
        "auth": "none",
        "downloadable": True,
        "notes": "toy descriptor for the B4 data-driven proof",
    }
    base.update(overrides)
    return SiteDescriptor(**base)


class TestEngineIsDataDriven:
    def test_toy_site_runs_only_its_declared_chain(self, tmp_path):
        from unittest.mock import patch

        connector = SimpleNamespace(stream_url=lambda url: {
            "url": "https://cdn.exampletunes.test/song.mp3",
            "protocol": "progressive", "mime_type": "audio/mpeg",
            "track": {"title": "Toy Song",
                      "permalink_url":
                          "https://exampletunes.test/toy-song"}})
        registry = SiteDescriptorRegistry([_toy_descriptor()])
        dl = _downloader(tmp_path, site_descriptors=registry,
                         site_connectors={"exampletunes": connector})
        # Direct must never run for this site: if it does, it dies.
        with patch("nomorals.tools.media.download",
                   side_effect=AssertionError("direct ran")):
            report = dl.download(
                "https://exampletunes.test/toy-song",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
        assert report.extractor == "exampletunes-connector"
        assert report.tried() == ["connector"]
        assert Path(report.path).read_bytes() == b"connector-bytes"

    def test_reordered_chain_runs_connector_first(self, tmp_path):
        from unittest.mock import patch

        connector = SimpleNamespace(stream_url=lambda url: {
            "url": "https://cdn.exampletunes.test/song.mp3",
            "protocol": "progressive", "mime_type": "audio/mpeg",
            "track": {"title": "Toy Song",
                      "permalink_url":
                          "https://exampletunes.test/toy-song"}})
        desc = _toy_descriptor(chain=("connector", "direct"))
        registry = SiteDescriptorRegistry([desc])
        dl = _downloader(tmp_path, site_descriptors=registry,
                         site_connectors={"exampletunes": connector})
        with patch("nomorals.tools.media.download",
                   side_effect=AssertionError("direct ran")):
            report = dl.download(
                "https://exampletunes.test/toy-song",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
        assert report.tried() == ["connector"]

    def test_chain_without_connector_never_engages_it(self, tmp_path):
        from unittest.mock import patch

        connector = SimpleNamespace(
            stream_url=mock.Mock(side_effect=AssertionError("engaged")))
        # SoundCloud's chain, minus the connector rung: the
        # injected connector must stay untouched.
        desc = replace(SOUNDCLOUD_DESCRIPTOR, chain=(
            "direct", "kit", "proxy", "rendered", "browser"))
        registry = SiteDescriptorRegistry([desc])
        dl = _downloader(tmp_path, connector,
                         site_descriptors=registry)
        with patch("nomorals.tools.media.download",
                   side_effect=_fail_direct), patch.object(
                       dl, "_stage_browser", return_value=False):
            report = dl.download(
                "https://soundcloud.com/artist/track",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        assert "connector" not in report.tried()
        connector.stream_url.assert_not_called()

    def test_not_downloadable_site_refused_with_its_reason(
            self, tmp_path):
        from unittest.mock import patch

        desc = _toy_descriptor(
            chain=CHAIN_STAGES, downloadable=False,
            notes="this toy site serves no downloads")
        registry = SiteDescriptorRegistry([desc])
        dl = _downloader(tmp_path, site_descriptors=registry)
        with patch("nomorals.tools.media.download",
                   side_effect=AssertionError("chain ran")):
            report = dl.download(
                "https://exampletunes.test/toy-song",
                dest_dir=tmp_path)
        assert report.ok is False
        assert report.extractor == "exampletunes"
        assert "exampletunes" in report.likely_cause
        assert report.tried() == ["extractor"]

    def test_auth_requirement_is_enforced(self, tmp_path, monkeypatch):
        # B5 honours the auth B4 only declared (RE-PINNED: the old
        # test asserted the requirement was data, not a gate).
        # A descriptor with auth="account" and no session stored
        # fails as found-but-blocked, class login, before the
        # chain runs — the connector that could have served the
        # track is never called.
        from unittest.mock import patch

        # The lazy resolver opens the home estate; keep the test
        # hermetic.
        monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
        calls: list[str] = []

        def stream_url(url):
            calls.append(url)
            return {
                "url": "https://cdn.exampletunes.test/song.mp3",
                "protocol": "progressive", "mime_type": "audio/mpeg",
                "track": {"title": "Toy Song",
                          "permalink_url":
                              "https://exampletunes.test/toy-song"}}

        connector = SimpleNamespace(stream_url=stream_url)
        desc = _toy_descriptor(
            auth="account", auth_domains=("exampletunes.test",))
        assert desc.auth_needed is True
        registry = SiteDescriptorRegistry([desc])
        dl = _downloader(tmp_path, site_descriptors=registry,
                         site_connectors={"exampletunes": connector})
        with patch("nomorals.tools.media.download",
                   side_effect=AssertionError("direct ran")):
            report = dl.download(
                "https://exampletunes.test/toy-song",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok is False
        assert report.block_reason == "login"
        assert report.tried() == ["auth"]
        assert calls == []
        assert "vault" in report.hint
