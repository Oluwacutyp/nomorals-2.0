"""Tests for nomorals/music_sources — legal-only registry + enrichment.

All offline: enrichment uses injected fakes, never real services.
"""

from pathlib import Path

import pytest

import nomorals.music_sources.registry as reg
from nomorals.music_sources import EnrichmentChain, MusicSource
from nomorals.music_sources.registry import (
    DEFAULT_PACK,
    MUSIC_REGISTRY,
    VOLATILITY_HEALTH_HOURS,
    all_sources,
    by_class,
    by_id,
    by_licence,
    by_pack,
    health_due,
    validate,
)

FORBIDDEN = ["pirated", "torrent", "leak", "rip"]


def _make_source(volatility="commercial", **overrides):
    kwargs = {
        "id": "t",
        "name": "T",
        "source_class": "metadata",
        "auth": "none",
        "formats": [],
        "quality_ceiling": "metadata only",
        "licence": "Open (MIT-style)",
        "volatility": volatility,
        "bulk": False,
        "api": False,
        "bot_wall": False,
        "etiquette": "Fetch politely.",
        "notes": "Test row.",
    }
    kwargs.update(overrides)
    return MusicSource(**kwargs)


# ── registry loads ───────────────────────────────────────────────────


class TestRegistryLoads:
    def test_count_at_least_60(self):
        assert len(all_sources()) >= 60
        assert len(MUSIC_REGISTRY) == len(all_sources())

    def test_all_ids_unique(self):
        ids = [s.id for s in all_sources()]
        assert len(ids) == len(set(ids))

    def test_all_rows_are_music_source(self):
        assert all(isinstance(s, MusicSource) for s in all_sources())

    def test_all_classes_present(self):
        classes = {s.source_class for s in all_sources()}
        for expected in ("purchase", "pool", "open", "artist_enabled",
                         "metadata", "preview", "dataset"):
            assert expected in classes

    def test_known_rows_present(self):
        ids = {s.id for s in all_sources()}
        for expected in ("qobuz", "bandcamp", "djcity", "traxsource",
                         "free_music_archive", "musicbrainz", "deezer",
                         "mtg_jamendo", "common_voice", "waxal",
                         "naijavoices", "boomplay", "ototo", "genie"):
            assert expected in ids


# ── legal-only ───────────────────────────────────────────────────────


class TestLegalOnly:
    def test_all_rows_have_nonempty_licence(self):
        for source in all_sources():
            assert isinstance(source.licence, str)
            assert source.licence != "" and not source.licence.isspace(), \
                source.id

    def test_no_forbidden_keywords_in_rows(self):
        for source in all_sources():
            blob = " ".join([source.id, source.name, source.notes]).lower()
            for word in FORBIDDEN:
                assert word not in blob, (source.id, word)

    def test_validate_rejects_missing_licence(self):
        with pytest.raises(ValueError):
            _make_source(licence="")
        with pytest.raises(ValueError):
            _make_source(licence="   ")
        good = _make_source()
        good.licence = ""
        with pytest.raises(ValueError):
            validate(good)

    def test_validate_rejects_invalid_class(self):
        with pytest.raises(ValueError):
            _make_source(source_class="bogus")
        good = _make_source()
        good.source_class = "bogus"
        with pytest.raises(ValueError):
            validate(good)

    def test_validate_rejects_invalid_volatility(self):
        with pytest.raises(ValueError):
            _make_source(volatility="bogus")
        good = _make_source()
        good.volatility = "bogus"
        with pytest.raises(ValueError):
            validate(good)

    def test_validate_accepts_good_row(self):
        assert validate(_make_source()) is not None


# ── lookups ──────────────────────────────────────────────────────────


class TestLookups:
    def test_by_id_found(self):
        found = by_id("qobuz")
        assert found is not None and found.name == "Qobuz"

    def test_by_id_missing_returns_none(self):
        assert by_id("no-such-source") is None

    def test_by_class_purchase(self):
        ids = {s.id for s in by_class("purchase")}
        assert "qobuz" in ids and "bandcamp" in ids
        assert "djcity" not in ids

    def test_by_class_pool(self):
        ids = {s.id for s in by_class("pool")}
        assert "djcity" in ids and "traxsource" in ids

    def test_by_class_dataset_includes_speech(self):
        ids = {s.id for s in by_class("dataset")}
        assert "mtg_jamendo" in ids and "common_voice" in ids

    def test_by_pack_nation(self):
        ids = {s.id for s in by_pack("nation", "JP")}
        assert "ototo" in ids and "jvs" in ids

    def test_by_pack_star_matches_all(self):
        assert len(by_pack("nation", "*")) == len(all_sources())
        assert len(by_pack("language", "*")) == len(all_sources())

    def test_by_pack_wildcard_row_matches_any_query(self):
        # qobuz pack nation is "*" (universal) so it matches any query.
        ids = {s.id for s in by_pack("nation", "ZZ")}
        assert "qobuz" in ids

    def test_by_licence(self):
        ids = {s.id for s in by_licence("DRM-free")}
        assert "qobuz" in ids and "bandcamp" in ids

    def test_by_licence_case_insensitive(self):
        assert {s.id for s in by_licence("cc")} == \
            {s.id for s in by_licence("CC")}
        assert "jamendo" in {s.id for s in by_licence("cc")}


# ── health cadence ────────────────────────────────────────────────────


class TestHealthDue:
    def test_health_hours_by_volatility(self):
        assert VOLATILITY_HEALTH_HOURS == {
            "institutional": 720,
            "commercial": 168,
            "indie": 24,
            "platform_api": 6,
        }
        assert _make_source("platform_api").health_check_hours == 6
        assert _make_source("indie").health_check_hours == 24
        assert _make_source("commercial").health_check_hours == 168
        assert _make_source("institutional").health_check_hours == 720

    def test_health_due_respects_volatility_ordering(self):
        now = 10_000_000.0
        # 7h ago: only platform_api (6h) is due.
        assert health_due(_make_source("platform_api"), now - 7 * 3600, now)
        assert not health_due(_make_source("indie"), now - 7 * 3600, now)
        # 25h ago: indie (24h) due, commercial (168h) not.
        assert health_due(_make_source("indie"), now - 25 * 3600, now)
        assert not health_due(_make_source("commercial"), now - 25 * 3600, now)
        # 169h ago: commercial due, institutional (720h) not.
        assert health_due(_make_source("commercial"), now - 169 * 3600, now)
        assert not health_due(_make_source("institutional"),
                               now - 169 * 3600, now)
        # 721h ago: institutional due.
        assert health_due(_make_source("institutional"),
                          now - 721 * 3600, now)

    def test_health_due_never_checked_is_due(self):
        assert health_due(_make_source(), None, 10_000_000.0)

    def test_health_due_future_check_not_due(self):
        assert not health_due(_make_source(), 10_000_001.0, 10_000_000.0)


# ── enrichment ────────────────────────────────────────────────────────


def _full_chain():
    return EnrichmentChain(
        fingerprint_fn=lambda x: "fp-123",
        acoustid_fn=lambda fp: {"acoustid_id": "ac-1",
                                "musicbrainz_id": "mb-1"},
        musicbrainz_fn=lambda mb: {"musicbrainz_id": "mb-1",
                                   "title": "Test Song"},
        discogs_fn=lambda mb: {"discogs_id": "d-1"},
        audiodb_fn=lambda mb: {"audiodb_id": "a-1"},
        coverart_fn=lambda mb: "https://cover.example.com/mb-1.jpg",
        acousticbrainz_fn=lambda mb: {"bpm": 120},
    )


class TestEnrichment:
    def test_full_chain_resolves(self):
        result = _full_chain().resolve("some/path.mp3")
        assert result.fingerprint == "fp-123"
        assert result.acoustid_id == "ac-1"
        assert result.musicbrainz_id == "mb-1"
        assert result.discogs == {"discogs_id": "d-1"}
        assert result.audiodb == {"audiodb_id": "a-1"}
        assert result.cover_art == "https://cover.example.com/mb-1.jpg"
        assert result.acousticbrainz == {"bpm": 120}
        assert result.missing == []
        assert all(step["status"] == "ok" for step in result.steps)
        assert {step["step"] for step in result.steps} == {
            "fingerprint", "acoustid", "musicbrainz", "discogs",
            "audiodb", "cover_art", "acousticbrainz"}

    def test_missing_steps_recorded_not_raised(self):
        result = EnrichmentChain().resolve("x.mp3")
        assert result.missing == ["fingerprint", "acoustid", "musicbrainz",
                                  "discogs", "audiodb", "cover_art",
                                  "acousticbrainz"]
        assert len(result.steps) == 7
        assert all(step["status"] == "missing"
                   for step in result.steps)

    def test_partial_chain_missing_recorded(self):
        chain = EnrichmentChain(fingerprint_fn=lambda x: "fp-9")
        result = chain.resolve("x.mp3")
        assert result.fingerprint == "fp-9"
        assert "acoustid" in result.missing
        assert "musicbrainz" in result.missing
        assert "discogs" in result.missing

    def test_failing_callable_recorded_not_raised(self):
        def boom(_x):
            raise RuntimeError("service down")

        chain = EnrichmentChain(fingerprint_fn=boom)
        result = chain.resolve("x.mp3")
        assert "fingerprint" in result.missing
        assert result.steps[0]["status"] == "error"

    def test_acoustid_plain_string(self):
        chain = EnrichmentChain(
            fingerprint_fn=lambda x: "fp-1",
            acoustid_fn=lambda fp: "ac-9",
            musicbrainz_fn=lambda mb: "mb-9",
        )
        result = chain.resolve("x.mp3")
        assert result.acoustid_id == "ac-9"
        assert result.musicbrainz_id == "mb-9"

    def test_resolve_batch(self):
        results = _full_chain().resolve_batch(["a.mp3", "b.mp3"])
        assert len(results) == 2
        assert all(r.musicbrainz_id == "mb-1" for r in results)


# ── pack / docstring generality ───────────────────────────────────────


class TestGenerality:
    def test_tone_language_docstring_generality(self):
        doc = reg.__doc__ or ""
        for word in ("Mandarin", "Cantonese", "Vietnamese", "Thai",
                     "Lao", "Hmong", "West African"):
            assert word in doc, word

    def test_default_pack_is_wildcard_not_a_nation(self):
        assert DEFAULT_PACK == {"nation": "*", "language": "*"}
        assert all(value == "*" for value in DEFAULT_PACK.values())

    def test_no_nomorals_imports_in_package(self):
        pkg = Path(reg.__file__).parent
        for path in pkg.glob("*.py"):
            text = path.read_text()
            assert "from nomorals" not in text, path
            assert "import nomorals" not in text, path

    def test_all_packs_are_dicts(self):
        for source in all_sources():
            assert isinstance(source.pack, dict)
