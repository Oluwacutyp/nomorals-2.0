"""Failing-first tests: the free-learning collection (OpenCulture,
52-websites site #12).

The collection is a curated layer over the sources registry:
CollectionEntry{subject, title, provider, url, format, level,
checked_at} seeded from the open directories, health-checked by the
sourcehealth idiom, and answered vetted-first with live search
second and clearly separated.
"""

import json
from types import SimpleNamespace

from nomorals.sources.learning import (
    SEED_DATE,
    SEED_ENTRIES,
    CollectionEntry,
    LearningCollection,
    LiveLearningSearch,
    check_entry,
    format_learning_reply,
    learn_free,
    refresh_health,
)


def _probe(status=200, body=b"", error=""):
    return SimpleNamespace(status=status, body=body, headers={},
                            error=error)


def _live_page():
    return (
        b'<html><body><a href="https://www.openculture.com/2026/01/01/'
        b'free-philosophy-course">Free Philosophy Course</a>'
        b'<a href="https://www.openculture.com/freeonlinecourses">'
        b'Free Online Courses</a></body></html>')


def _entry(**kw):
    base = {
        "subject": "economics", "title": "Test Course",
        "provider": "Test U", "url": "https://example.test/course",
        "format": "course", "level": "all",
        "description": "A test course about economics.",
        "provenance": "Test directory", "seeded_at": SEED_DATE,
    }
    base.update(kw)
    return CollectionEntry(**base)


class _NoLive(LiveLearningSearch):
    """Live rung that finds nothing — the collection answers alone."""

    def search_all(self, query, *, limit=5):
        return []


# ── seeds ──────────────────────────────────────────────────────────────────

def test_seed_collection_count_and_subjects():
    assert len(SEED_ENTRIES) == 24
    subjects = {e.subject for e in SEED_ENTRIES}
    assert len(subjects) >= 10
    for wanted in ("economics", "history", "computer science",
                   "physics", "mathematics", "spanish", "french",
                   "german", "literature", "film", "psychology"):
        assert wanted in subjects


def test_seed_entries_have_provenance_and_seed_date():
    for entry in SEED_ENTRIES:
        assert entry.provenance, entry
        assert entry.seeded_at == SEED_DATE, entry
        assert entry.format in ("course", "textbook", "audiobook",
                                "ebook", "movie", "lesson")
        assert entry.level in ("beginner", "intermediate", "advanced",
                               "all")
        assert entry.url.startswith("https://"), entry
        assert entry.health_status == "unknown"
        assert entry.checked_at is None


def test_seed_slugs_unique():
    slugs = [e.slug for e in SEED_ENTRIES]
    assert len(slugs) == len(set(slugs))


# ── the query: vetted first ────────────────────────────────────────────────

def test_subject_query_returns_matching_entries():
    coll = LearningCollection()
    result = learn_free(coll, "economics", live_search=_NoLive())
    assert result["ok"] is True
    assert result["subject"] == "economics"
    titles = [e["title"] for e in
              result["vetted"] + result["unchecked"]]
    assert "Financial Markets" in titles
    assert "Game Theory" in titles
    assert "Principles of Microeconomics" in titles
    # Every served entry carries provenance + a check-date field.
    for band in ("vetted", "unchecked", "flagged", "live"):
        for entry in result[band]:
            assert entry["provenance"], entry
            assert "checked_date" in entry, entry


def test_query_normalization_strips_learn_free_framing():
    coll = LearningCollection()
    result = learn_free(coll, "I want to learn economics, free",
                        live_search=_NoLive())
    assert result["subject"] == "economics"
    assert result["unchecked"], "normalised query must still match"


def test_vetted_first_unchecked_labelled_flagged_separate():
    coll = LearningCollection()
    now = 1784300000.0
    live = _entry(title="Live Course",
                  url="https://example.test/live",
                  health_status="verified-live", checked_at=now)
    fresh = _entry(title="Fresh Course",
                   url="https://example.test/fresh",
                   health_status="verified-live",
                   checked_at=now - 60 * 86400)
    unchecked = _entry(title="Unchecked Course",
                       url="https://example.test/unchecked")
    dead = _entry(title="Dead Course",
                  url="https://example.test/dead",
                  health_status="dead", checked_at=now,
                  last_error="HTTP 404")
    for entry in (live, fresh, unchecked, dead):
        coll.entries[entry.slug] = entry
    result = learn_free(coll, "economics", live_search=_NoLive())
    assert [e["title"] for e in result["vetted"]] == \
        ["Fresh Course", "Live Course"]
    assert [e["title"] for e in result["unchecked"]] == \
        ["Financial Markets", "Game Theory",
         "Principles of Microeconomics", "Unchecked Course"]
    assert [e["title"] for e in result["flagged"]] == ["Dead Course"]
    # The reply shows the check date on vetted, "not yet checked" on
    # unchecked, and the reason on flagged — in that order.
    reply = format_learning_reply("economics", result)
    assert "Vetted — checked live:" in reply
    assert "not yet checked" in reply
    assert "Flagged — not recommended" in reply
    assert "HTTP 404" in reply
    assert reply.index("Vetted") < reply.index("not yet checked")
    assert reply.index("not yet checked") < reply.index("Flagged")


def test_dead_entry_flagged_not_served_as_good():
    coll = LearningCollection()
    dead = _entry(title="Dead Course",
                  url="https://example.test/dead",
                  health_status="dead", checked_at=1784300000.0,
                  last_error="HTTP 404")
    coll.entries[dead.slug] = dead
    result = learn_free(coll, "economics", live_search=_NoLive())
    assert "Dead Course" not in [e["title"] for e in result["vetted"]]
    assert "Dead Course" not in [e["title"] for e in result["unchecked"]]
    assert "Dead Course" in [e["title"] for e in result["flagged"]]
    assert result["flagged"][0]["last_error"] == "HTTP 404"


def test_format_and_level_filters():
    coll = LearningCollection()
    textbooks = learn_free(coll, "", format="textbook",
                           live_search=_NoLive())
    served = textbooks["vetted"] + textbooks["unchecked"]
    assert served
    assert all(e["format"] == "textbook" for e in served)
    assert any(e["title"] == "College Physics" for e in served)
    beginners = learn_free(coll, "physics", level="beginner",
                           live_search=_NoLive())
    served = beginners["vetted"] + beginners["unchecked"]
    assert any(e["title"] == "College Physics" for e in served)
    assert all(e["title"] != "University Physics Volume 1"
               for e in served)


def test_empty_subject_returns_everything_banded():
    coll = LearningCollection()
    result = learn_free(coll, "", live_search=_NoLive())
    assert result["unchecked"], "empty subject lists the collection"


def test_no_match_answers_honestly():
    coll = LearningCollection()
    result = learn_free(coll, "quantum chromodynamics knitting",
                        live_search=_NoLive())
    assert result["vetted"] == []
    assert result["unchecked"] == []
    assert result["flagged"] == []
    assert result["live"] == []
    reply = format_learning_reply("quantum chromodynamics knitting",
                                  result)
    assert "Nothing found" in reply


# ── health ─────────────────────────────────────────────────────────────────

def test_check_entry_verified_live():
    entry = _entry()
    fresh = check_entry(
        entry,
        fetcher=lambda url: _probe(
            200, b'<a href="https://example.test/a">A</a>'
                  b'<a href="https://example.test/b">B</a>'),
        now=1794960000.0)
    assert fresh.health_status == "verified-live"
    assert fresh.checked_at == 1794960000.0
    assert fresh.checked_date == "2026-11-18"
    assert fresh.last_error == ""


def test_check_entry_dead_on_404():
    entry = _entry()
    fresh = check_entry(entry, fetcher=lambda url: _probe(404, b"gone"),
                        now=1784300000.0)
    assert fresh.health_status == "dead"
    assert fresh.last_error == "HTTP 404"


def test_check_entry_dead_on_empty_200():
    entry = _entry()
    fresh = check_entry(entry, fetcher=lambda url: _probe(200, b"<html></html>"),
                        now=1784300000.0)
    assert fresh.health_status == "dead"
    assert "zero usable links" in fresh.last_error


def test_check_entry_bot_blocked():
    entry = _entry()
    fresh = check_entry(
        entry,
        fetcher=lambda url: _probe(
            200, b"<html>Please verify you are human</html>"),
        now=1784300000.0)
    assert fresh.health_status == "bot-blocked"


def test_check_entry_unverifiable_on_transport_failure():
    entry = _entry()
    fresh = check_entry(entry, fetcher=lambda url: _probe(0, b"",
                                                          error="boom"),
                        now=1784300000.0)
    assert fresh.health_status == "unverifiable"
    assert fresh.last_error == "boom"


def test_check_entry_never_raises():
    entry = _entry()

    def _boom(url):
        raise RuntimeError("no network here")

    fresh = check_entry(entry, fetcher=_boom, now=1784300000.0)
    assert fresh.health_status == "unverifiable"


def test_refresh_health_only_rechecks_stale():
    coll = LearningCollection()
    # The 24 seeds are all unchecked, so they would be re-probed too —
    # clear them: this test is about fresh-vs-stale, not the seeds.
    coll.entries = {}
    now = 1784300000.0
    fresh = _entry(title="Fresh Course",
                   url="https://example.test/fresh",
                   health_status="verified-live", checked_at=now - 3600)
    stale = _entry(title="Stale Course",
                   url="https://example.test/stale",
                   health_status="verified-live",
                   checked_at=now - 8 * 86400)
    dead = _entry(title="Dead Course",
                  url="https://example.test/dead",
                  health_status="unknown", checked_at=None)
    for entry in (fresh, stale, dead):
        coll.entries[entry.slug] = entry
    calls: list[str] = []

    def _fetcher(url):
        calls.append(url)
        if "stale" in url:
            return _probe(200, b'<a href="https://example.test/x">X</a>')
        return _probe(404, b"gone")

    summary = refresh_health(coll, fetcher=_fetcher, now=now)
    assert summary["checked"] == 2
    assert summary["live"] == 1
    assert dead.slug in summary["flagged"]
    assert "https://example.test/fresh" not in calls
    assert coll.entries[stale.slug].checked_at == now
    assert coll.entries[dead.slug].health_status == "dead"


# ── persistence ────────────────────────────────────────────────────────────

def test_collection_persists_and_grows(tmp_path):
    coll = LearningCollection(tmp_path)
    assert len(coll.all_entries()) == len(SEED_ENTRIES)
    assert (tmp_path / "learning_collection.json").exists()
    coll.upsert(_entry(title="New Course",
                       url="https://example.test/new"))
    again = LearningCollection(tmp_path)
    assert any(e.title == "New Course" for e in again.all_entries())
    assert again.entry(_entry(title="New Course",
                              url="https://example.test/new").slug) is not None


def test_corrupt_store_reseeds(tmp_path):
    path = tmp_path / "learning_collection.json"
    path.write_text("{not json", encoding="utf-8")
    coll = LearningCollection(tmp_path)
    assert len(coll.all_entries()) == len(SEED_ENTRIES)


def test_subjects_listed(tmp_path):
    coll = LearningCollection(tmp_path)
    subjects = coll.subjects()
    assert "economics" in subjects
    assert subjects == sorted(subjects)


# ── live search ────────────────────────────────────────────────────────────

def test_live_search_openculture_parse():
    searcher = LiveLearningSearch(
        fetcher=lambda url: _probe(200, _live_page()))
    found = searcher.search_openculture("philosophy")
    assert len(found) == 1
    assert found[0].title == "Free Philosophy Course"
    assert found[0].provider == "OpenCulture"
    assert found[0].provenance == "OpenCulture site search"
    assert found[0].health_status == "unknown"


def test_live_search_openculture_drops_site_chrome():
    # The search page also carries the site's own section links
    # ("1,150 Free Movies" ...). They never carry the query, so the
    # relevance gate drops them — only real results survive.
    page = (
        b'<html><body><a href="https://www.openculture.com/freemoviesonline">'
        b'1,150 Free Movies</a>'
        b'<a href="https://www.openculture.com/2026/01/01/'
        b'free-philosophy-course">Free Philosophy Course</a>'
        b'<a href="https://www.openculture.com/freeaudiobooks">'
        b'1,000 Free Audio Books</a></body></html>')
    searcher = LiveLearningSearch(
        fetcher=lambda url: _probe(200, page))
    found = searcher.search_openculture("philosophy")
    assert [f.title for f in found] == ["Free Philosophy Course"]


def test_live_search_archive_parse():
    body = json.dumps({
        "response": {"docs": [
            {"identifier": "prideprejudice123",
             "title": "Pride and Prejudice", "creator": "Jane Austen"},
            {"identifier": "", "title": "No identifier"},
        ]}}).encode()
    searcher = LiveLearningSearch(
        fetcher=lambda url: _probe(200, body))
    found = searcher.search_archive("pride and prejudice")
    assert len(found) == 1
    assert found[0].url == \
        "https://archive.org/details/prideprejudice123"
    assert "Jane Austen" in found[0].provider
    assert found[0].provenance == "Archive.org advanced search"


def test_live_search_openlibrary_parse():
    body = json.dumps({
        "docs": [
            {"title": "Pride and Prejudice", "key": "/works/OL123W",
             "author_name": ["Jane Austen"]},
            {"title": "No key here"},
        ]}).encode()
    searcher = LiveLearningSearch(
        fetcher=lambda url: _probe(200, body))
    found = searcher.search_openlibrary("pride and prejudice")
    assert len(found) == 1
    assert found[0].url == "https://openlibrary.org/works/OL123W"
    assert found[0].provenance == "Open Library search"


def test_live_search_never_raises():
    def _boom(url):
        raise RuntimeError("down")

    searcher = LiveLearningSearch(fetcher=_boom)
    assert searcher.search_openculture("x") == []
    assert searcher.search_archive("x") == []
    assert searcher.search_openlibrary("x") == []
    assert searcher.search_all("x") == []


def test_learn_free_live_second_and_separated():
    coll = LearningCollection()
    live_hit = CollectionEntry(
        subject="astrophysics", title="Live Only Course",
        provider="OpenCulture", url="https://example.test/live-only",
        format="course", level="all",
        description="Found by OpenCulture site search.",
        provenance="OpenCulture site search",
        seeded_at=SEED_DATE)

    class _OneHit(LiveLearningSearch):
        def search_all(self, query, *, limit=5):
            return [live_hit]

    result = learn_free(coll, "astrophysics", live_search=_OneHit())
    assert result["vetted"] == []
    assert result["unchecked"] == []
    assert [e["title"] for e in result["live"]] == ["Live Only Course"]
    reply = format_learning_reply("astrophysics", result)
    assert "Live search (not vetted yet):" in reply
    assert "Live Only Course" in reply


def test_learn_free_live_hit_duplicating_seed_is_dropped():
    coll = LearningCollection()
    seed = SEED_ENTRIES[0]
    dup = CollectionEntry(
        subject="economics", title=seed.title, provider="OpenCulture",
        url=seed.url, format="course", level="all",
        description="Found by OpenCulture site search.",
        provenance="OpenCulture site search", seeded_at=SEED_DATE)

    class _DupHit(LiveLearningSearch):
        def search_all(self, query, *, limit=5):
            return [dup]

    result = learn_free(coll, "economics", live_search=_DupHit())
    live_urls = [e["url"] for e in result["live"]]
    assert seed.url not in live_urls


# ── registration: source family + brain tool ───────────────────────────────

def test_registry_collects_learning_family():
    from nomorals.agency.sources_registry import collect, spec_by_key

    specs, errors = collect()
    assert errors == []
    cats = {s.category for s in specs}
    assert "learning" in cats
    learning = [s for s in specs if s.category == "learning"]
    assert len(learning) == 5
    spec = spec_by_key(specs, "learning:openculture-courses")
    assert spec is not None
    assert spec.url == "https://www.openculture.com/freeonlinecourses"
    assert spec.kind == "html-links"
    for entry in learning:
        assert entry.url.startswith("https://"), entry


def test_brain_tool_registered():
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_builtins()
    assert "free_learning" in reg._tools
    outcome = reg.call("free_learning", actor="test",
                       subject="economics", limit=3)
    assert outcome.ok, outcome.error
    result = outcome.value
    assert result["ok"] is True
    titles = [e["title"] for e in
              result["vetted"] + result["unchecked"]]
    assert "Financial Markets" in titles
    assert isinstance(result["live"], list)
