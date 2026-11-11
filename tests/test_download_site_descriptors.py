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
    BLOCK_CLASSES,
    CHAIN_STAGES,
    SOUNDCLOUD_DESCRIPTOR,
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
            "downloadable": True, "notes": "toy",
        })
        assert desc.site == "tunes"
        assert desc.auth_needed is True
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

    def test_auth_requirement_is_declared_not_enforced(self, tmp_path):
        # B4 declares auth; B5 is the rung that honours it.  A
        # descriptor with auth="account" still runs its chain on
        # the access the chain already has — the requirement is
        # data, not a new gate.
        from unittest.mock import patch

        connector = SimpleNamespace(stream_url=lambda url: {
            "url": "https://cdn.exampletunes.test/song.mp3",
            "protocol": "progressive", "mime_type": "audio/mpeg",
            "track": {"title": "Toy Song",
                      "permalink_url":
                          "https://exampletunes.test/toy-song"}})
        desc = _toy_descriptor(auth="account")
        assert desc.auth_needed is True
        registry = SiteDescriptorRegistry([desc])
        dl = _downloader(tmp_path, site_descriptors=registry,
                         site_connectors={"exampletunes": connector})
        with patch("nomorals.tools.media.download",
                   side_effect=AssertionError("direct ran")):
            report = dl.download(
                "https://exampletunes.test/toy-song",
                dest_dir=tmp_path, audio_only=True)
        assert report.ok
