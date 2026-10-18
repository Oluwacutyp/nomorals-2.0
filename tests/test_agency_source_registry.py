"""Failing-first tests: the source registry is single-sourced.

Every verifiable source list in the repo is collected from its REAL
module — never a duplicated URL list in the registry. If a list moves,
the registry follows it. A registry entry without a checkable URL is a
bug, except for the documented adapter-gated kinds.
"""

import pytest

from nomorals.agency.sources_registry import collect, spec_by_key


def test_collect_covers_every_source_family():
    specs, errors = collect()
    by_cat = {}
    for s in specs:
        by_cat.setdefault(s.category, []).append(s)
    for cat in ("news", "search", "music", "film", "weather", "wisdom",
                "price"):
        assert cat in by_cat, f"missing category {cat} (errors={errors})"
        assert len(by_cat[cat]) >= 1
    assert errors == [], f"registry imports failed: {errors}"


def test_news_specs_match_the_real_module_exactly():
    from nomorals.agents.news import FEED_CATEGORIES

    specs, _ = collect()
    news_urls = {s.url for s in specs if s.category == "news"}
    real_urls = {url for vals in FEED_CATEGORIES.values() for _, url in vals}
    assert news_urls == real_urls


def test_search_specs_match_the_real_engine_list():
    from nomorals.tools.web import _SEARCH_ENGINES

    specs, _ = collect()
    names = {s.name for s in specs if s.category == "search"}
    assert names == set(_SEARCH_ENGINES)


def test_music_specs_match_the_real_chain():
    from nomorals.media.sources import SOURCE_CHAIN

    specs, _ = collect()
    names = {s.name for s in specs if s.category == "music"}
    chain_names = {name for name, _ in SOURCE_CHAIN}
    # The chain is the registry's music set (boomplay removed: JS SPA,
    # plain-HTTP search returns no results — verified 2026-10-18).
    assert names == chain_names
    assert "boomplay-search" not in names


def test_no_duplicate_urls_inside_a_category():
    specs, _ = collect()
    for cat in ("news", "search", "music", "film", "weather", "wisdom",
                "price"):
        urls = [s.url for s in specs if s.category == cat]
        assert len(urls) == len(set(urls)), f"dup URLs in {cat}"


def test_every_spec_has_a_checkable_url_or_adapter_note():
    specs, _ = collect()
    for s in specs:
        if s.kind == "external-adapter":
            assert s.notes, f"{s.key} adapter without notes"
        else:
            assert s.url.startswith(("http://", "https://")), s


def test_spec_by_key_roundtrip():
    specs, _ = collect()
    first = specs[0]
    assert spec_by_key(specs, first.key) is first
    assert spec_by_key(specs, "no-such-key") is None
