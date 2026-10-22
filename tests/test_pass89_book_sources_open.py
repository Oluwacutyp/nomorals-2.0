"""Pass #89 — book/story sources go open (the owner's open-sources law).

Passes #86/#87/#88 took downloads + music open; this suite pins the
last named CLOSED-SOURCE class from those reports: books/story sites.

Survey truth this suite pins:

* Book/story sources live in TWO homes in ``nomorals/books``:
  dedicated adapters in ``sources.py`` (freewebnovel, novelfull,
  royalroad, pandanovel-as-generic) and descriptor-as-data adapters
  in ``descriptors.py`` (gutenberg, scribblehub, ao3, wattpad,
  webnovel), plus a self-tuning ``GenericAdapter`` fallback for any
  unknown host.  The reader (``reader.py``) consumes them through
  ``adapter_for_url`` / ``search_all``; ``books/tools.py``
  ``story_download`` consumes ``download_story`` directly.  None of
  them recorded an outcome, held health state, admitted a new site,
  or remembered anything on restart.
* Architecture decision, with evidence: the media registry's model
  FITS the book class without distortion — every book source is one
  host serving novel/chapter fetches (the same (nation, language,
  domain) key, the same outcome/health/candidate vocabulary, the
  same KV persistence) — so book sources join the SAME
  ``media/source_registry.SourceRegistry`` as their own class
  (option (a) in the brief), exactly like music in #88.  No parallel
  stats store, no second health loop, no second scheduler job: the
  #86 ``download_sources`` health job covers the book class because
  it runs over the one registry.
* A bot-wall / Cloudflare fetch is NOT a dead source: adapters raise
  ``SourceBlocked`` (the #78/#79/#81 estate's seam), recorded with
  error class ``bot_block`` and health ``bot-blocked`` — the
  registry's existing health vocabulary — never ``dead``.
* "Site up, story missing" is weak evidence: an empty search /
  chapter list, and a landed page with no chapter body, are recorded
  via ``record_no_result`` (#88 precedent) — they sink the score but
  never trip the circuit breaker.
* Discovery channels that genuinely exist on the book path:
  redirect targets (a fetch that lands on a foreign host), story-
  shaped links embedded in fetched pages (a site linking its mirror
  / the same novel elsewhere — the Fetcher seam sees the raw page),
  and foreign hosts carried by fetch/search results themselves
  (chapter prev/next links, search hits).  Channels that do NOT
  exist: no sitemap/feed/discovery listing is fetched by any book
  consumer, and cover-image hosts are CDNs, not story sites (never
  intaken).  Candidates serve nothing: they are absent from
  ``search_all``'s pool and from the grown-adapter list until the
  health loop promotes them; a promoted source serves through the
  universal ``GenericAdapter`` shape (the descriptor reader's
  generic shape — the #88 ``DiscoveredBlogSource`` precedent).

All fetches are stubbed — no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


# ── helpers ───────────────────────────────────────────────────────────


def _kv(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _seeded(store=None):
    from nomorals.media.source_registry import _seed

    return _seed(store=store)


def _ctx(tmp_path, db=None):
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


def _dead_fetcher(url):
    return 404, b"gone"


from nomorals.books import sources as book_sources  # noqa: E402
from nomorals.books.sources import (  # noqa: E402
    FreeWebNovelAdapter, NovelFullAdapter, RoyalRoadAdapter,
    SourceBlocked, SourceError)
from nomorals.tools.browser import parse_html  # noqa: E402


class _FakeFetcher:
    """Serves canned pages instead of the network (the test_books_
    sources harness).  A value may be an exception to raise."""

    def __init__(self, pages):
        self.pages = pages

    def get(self, url, **_):
        # longest prefix first: a novel URL is a prefix of its
        # chapter URLs, so insertion order would mis-serve chapters
        for key in sorted(self.pages, key=len, reverse=True):
            html = self.pages[key]
            if url.startswith(key):
                if isinstance(html, Exception):
                    raise html
                return url, parse_html(html), html
        raise SourceError(f"no fixture for {url}")


FWN_SEARCH = """<html><body>
<a href="/novel/alpha-tale">Alpha Tale</a>
</body></html>"""

NF_SEARCH = """<html><body>
<a href="https://novelfull.net/beta-tale.html">Beta Tale</a>
</body></html>"""

RR_SEARCH = """<html><body>
<a href="https://www.royalroad.com/fiction/111/gamma-tale">Gamma Tale</a>
</body></html>"""

FWN_NOVEL = """<html><head>
<meta property="og:title" content="Alpha Tale"/>
<meta property="og:novel:author" content="Awespec"/>
<meta property="og:novel:status" content="OnGoing"/>
</head><body>
<a href="/novel/alpha-tale/chapter-1">Chapter 1</a>
<a href="/novel/alpha-tale/chapter-2">Chapter 2</a>
<a href="/novel/alpha-tale/chapter-3">Chapter 3</a>
</body></html>"""


def _fwn_chapter(n):
    body = " ".join(
        [f"Alpha chapter {n} prose marched onward through the gate."] * 6)
    return f"""<html><body><div class="txt"><div id="article">
<h4>Chapter {n}</h4><p>{body}</p><p>More prose follows here.</p>
</div></div>
<a href="/novel/alpha-tale/chapter-{n-1}">Prev Chapter</a>
<a href="/novel/alpha-tale/chapter-{n+1}">Next Chapter</a>
</body></html>"""


def _search_adapters():
    return [
        FreeWebNovelAdapter(fetcher=_FakeFetcher({
            "https://freewebnovel.com/search": FWN_SEARCH})),
        NovelFullAdapter(fetcher=_FakeFetcher({
            "https://novelfull.net/search": NF_SEARCH})),
        RoyalRoadAdapter(fetcher=_FakeFetcher({
            "https://www.royalroad.com/fictions/search": RR_SEARCH})),
    ]


# ── (1) the book class joins the SAME registry ────────────────────────


def test_every_book_source_is_a_registered_seed():
    from nomorals.books.sources import (
        BOOK_SOURCE_BY_DOMAIN, BOOK_SOURCE_DOMAINS)

    reg = _seeded()
    origins = {s["domain"]: s["origin"] for s in reg.status()["sources"]}
    for source_id, domain in BOOK_SOURCE_DOMAINS.items():
        desc = reg.for_domain(domain)
        assert desc is not None, f"{source_id} ({domain}) not registered"
        assert "book" in desc.kinds, (source_id, desc.kinds)
        assert origins[domain] == "seed", (source_id, origins[domain])
    # every known adapter/descriptor domain (mirrors included) maps
    # back to its source and is registered too
    for domain, source_id in BOOK_SOURCE_BY_DOMAIN.items():
        assert reg.for_domain(domain) is not None, domain


def test_book_seeds_do_not_change_the_extractor_view():
    from nomorals.media.download_kit import descriptors_for_registry

    names = [d.name for d in descriptors_for_registry(_seeded())]
    assert names == [
        "netnaija", "audiomack", "boomplay", "notjustok", "tooxclusive",
        "youtube", "soundcloud", "spotify", "tiktok", "instagram",
        "x-twitter", "direct-media", "ytdlp-generic"], names


# ── (2) day-one pin: no learned state → today's search order ──────────


def test_day_one_search_order_is_byte_for_byte(tmp_path):
    reg = _seeded()
    with mock.patch.object(book_sources, "ADAPTERS", _search_adapters()):
        plain = book_sources.search_all("tale", limit=10)
        learned = book_sources.search_all(
            "tale", limit=10, source_registry=reg)
    assert [h.source for h in plain] == [
        "freewebnovel", "novelfull", "royalroad"]
    assert [h.source for h in learned] == [h.source for h in plain]
    assert [h.url for h in learned] == [h.url for h in plain]


# ── (3) outcome recording on the real fetch paths ─────────────────────


def test_reader_fetch_paths_credit_the_serving_source(tmp_path):
    from nomorals.books import reader as reader_mod
    from nomorals.books.reader import StoryReader

    kv, db = _kv(tmp_path)
    reg = _seeded(store=kv)
    ctx = _ctx(tmp_path, db=db)
    pages = {"https://freewebnovel.com/novel/alpha-tale": FWN_NOVEL}
    for n in (1, 2, 3):
        pages[f"https://freewebnovel.com/novel/alpha-tale/chapter-{n}"] = \
            _fwn_chapter(n)
    adapter = FreeWebNovelAdapter(fetcher=_FakeFetcher(pages))
    reader = StoryReader(ctx, source_registry=reg)
    with mock.patch.object(reader_mod, "adapter_for_url",
                           lambda url, **kw: adapter):
        story = reader.follow("https://freewebnovel.com/novel/alpha-tale")
        out = reader.read(story.slug, chapter=1)
    assert "Alpha chapter 1" in out["text"]
    desc = reg.for_domain("freewebnovel.com")
    st = reg.stats_for(desc)
    # novel (follow) + chapter_list (_chapter_url) + fetch_chapter
    assert st["ok"] >= 3 and st["fail"] == 0, st
    # reader invariants on this very path: progress moved, no more
    prog = reader.get_progress(story.slug)
    assert prog["chapter"] == 1


def test_follow_ingest_all_chapters_records_and_completes(tmp_path):
    from nomorals.books import reader as reader_mod
    from nomorals.books.reader import StoryReader

    kv, db = _kv(tmp_path)
    reg = _seeded(store=kv)
    ctx = _ctx(tmp_path, db=db)
    pages = {"https://freewebnovel.com/novel/alpha-tale": FWN_NOVEL}
    for n in (1, 2, 3):
        pages[f"https://freewebnovel.com/novel/alpha-tale/chapter-{n}"] = \
            _fwn_chapter(n)
    adapter = FreeWebNovelAdapter(fetcher=_FakeFetcher(pages))
    reader = StoryReader(ctx, source_registry=reg)
    with mock.patch.object(reader_mod, "adapter_for_url",
                           lambda url, **kw: adapter), \
            mock.patch.object(StoryReader, "_profile",
                              lambda self: {"sync_chapters": 3,
                                            "list_limit": 60,
                                            "pause_s": 0.0,
                                            "timeout": 5.0}):
        story = reader.follow("https://freewebnovel.com/novel/alpha-tale")
        report = reader.ingest(story.slug)
    # BookForge invariant: follow-ingest pulls ALL chapters and
    # never moves the reading position
    assert report["complete"] and report["ingested"] == 3, report
    assert reader.get_progress(story.slug)["chapter"] == 0
    st = reg.stats_for(reg.for_domain("freewebnovel.com"))
    assert st["ok"] >= 5 and st["fail"] == 0, st


def test_fetch_failure_records_error_class_and_sinks_score(tmp_path):
    reg = _seeded()
    pristine = _seeded()
    adapter = FreeWebNovelAdapter(fetcher=_FakeFetcher({
        "https://freewebnovel.com/novel/x/chapter-1":
            SourceError("https://freewebnovel.com unreachable: reset")}))
    adapter.bind_book_registry(reg)
    with pytest.raises(SourceError):
        adapter.fetch_chapter("https://freewebnovel.com/novel/x/chapter-1")
    desc = reg.for_domain("freewebnovel.com")
    st = reg.stats_for(desc)
    assert st["fail"] == 1 and st["last_error_class"] == "network_failure", st
    assert reg.score(desc) < pristine.score(
        pristine.for_domain("freewebnovel.com"))


def test_bot_wall_is_blocked_not_dead(tmp_path):
    reg = _seeded()
    adapter = FreeWebNovelAdapter(fetcher=_FakeFetcher({
        "https://freewebnovel.com/novel/x/chapter-1":
            SourceBlocked("https://freewebnovel.com served a bot-wall page")}))
    adapter.bind_book_registry(reg)
    with pytest.raises(SourceBlocked):
        adapter.fetch_chapter("https://freewebnovel.com/novel/x/chapter-1")
    desc = reg.for_domain("freewebnovel.com")
    st = reg.stats_for(desc)
    assert st["blocked"] == 1, st
    assert st["last_error_class"] == "bot_block", st
    health = reg.health_for(desc)
    assert health == "bot-blocked", health  # NOT dead, NOT disabled


def test_no_result_is_weak_evidence_never_a_breaker_trip(tmp_path):
    reg = _seeded()
    pristine = _seeded()
    adapter = FreeWebNovelAdapter(fetcher=_FakeFetcher({
        "https://freewebnovel.com/search":
            "<html><body><p>nothing here</p></body></html>"}))
    adapter.bind_book_registry(reg)
    for _ in range(5):
        assert adapter.search("obscure tale xyz") == []
    desc = reg.for_domain("freewebnovel.com")
    st = reg.stats_for(desc)
    assert st["no_result"] == 5 and st["fail"] == 0, st
    assert reg.health_for(desc) == "unknown"  # breaker untouched
    assert reg.score(desc) < pristine.score(
        pristine.for_domain("freewebnovel.com"))


def test_landed_page_without_chapter_body_is_weak_evidence(tmp_path):
    reg = _seeded()
    adapter = FreeWebNovelAdapter(fetcher=_FakeFetcher({
        "https://freewebnovel.com/novel/x/chapter-1":
            "<html><body><div id='article'></div></body></html>"}))
    adapter.bind_book_registry(reg)
    with pytest.raises(SourceError):
        adapter.fetch_chapter("https://freewebnovel.com/novel/x/chapter-1")
    st = reg.stats_for(reg.for_domain("freewebnovel.com"))
    assert st["no_result"] == 1 and st["fail"] == 0, st


def test_unmapped_search_is_no_evidence(tmp_path):
    # GenericAdapter.search honestly raises "not mapped" — a fact
    # about the adapter, not about any source's health.
    from nomorals.books.sources import GenericAdapter

    reg = _seeded()
    adapter = GenericAdapter(name="generic:example.com",
                             domains=("example.com",),
                             fetcher=_FakeFetcher({}))
    adapter.bind_book_registry(reg)
    with pytest.raises(SourceError):
        adapter.search("anything")
    assert reg.status()["sources"]  # registry unharmed, no stats written
    assert all(s["ok"] == 0 and s["fail"] == 0
               for s in reg.status()["sources"])
    assert all(r["ok"] == 0 and r["fail"] == 0 and r["no_result"] == 0
               for r in reg.status()["books"]["sources"])


# ── (4) ordering becomes outcome-aware where a choice exists ──────────


def test_stood_down_source_is_demoted_in_search_order(tmp_path):
    reg = _seeded()
    desc = reg.for_domain("freewebnovel.com")
    for _ in range(3):
        reg.record_outcome(desc, False, error_class="network_failure")
    assert reg.health_for(desc) == "stood-down"
    with mock.patch.object(book_sources, "ADAPTERS", _search_adapters()):
        hits = book_sources.search_all(
            "tale", limit=10, source_registry=reg)
    order = [h.source for h in hits]
    assert order.index("freewebnovel") > order.index("royalroad"), order
    assert order[-1] == "freewebnovel", order  # demoted, still tried last


# ── (5) discovery intake on the book path ─────────────────────────────


def _real_fetcher(pages):
    """A real books Fetcher whose HTTP tier serves canned pages:
    {request_prefix: (final_url, html)}."""
    from nomorals.books.sources import Fetcher

    fetcher = Fetcher(rate_limit=0.0)

    def http_get(url, *, referer="", retries=0):
        for key, (final, html) in pages.items():
            if url.startswith(key):
                return final, html
        raise SourceError(f"no fixture for {url}")

    fetcher.http_get = http_get
    return fetcher


def test_page_embedded_story_link_is_noted_untrusted(tmp_path):
    reg = _seeded()
    html = """<html><body><div class="txt"><div id="article">
<h4>Chapter 1</h4><p>Prose one two three four five six seven.</p>
</div></div>
<a href="https://mirrorstories.example/novel/alpha-tale/chapter-1">
Read at our mirror</a>
<a href="https://freewebnovel.com/novel/alpha-tale/chapter-2">Next</a>
<a href="https://socialshare.example/share">Share</a>
</body></html>"""
    fetcher = _real_fetcher({
        "https://freewebnovel.com/novel/alpha-tale/chapter-1":
            ("https://freewebnovel.com/novel/alpha-tale/chapter-1", html)})
    adapter = FreeWebNovelAdapter(fetcher=fetcher)
    adapter.bind_book_registry(reg)
    ch = adapter.fetch_chapter(
        "https://freewebnovel.com/novel/alpha-tale/chapter-1")
    assert ch.number == 1
    cand_domains = [c.domain for c in reg.candidates()]
    # the story-shaped mirror link is intaken; the share link (no
    # novel/chapter shape) is not a story source and is ignored
    assert cand_domains == ["mirrorstories.example"], cand_domains
    assert reg.for_domain("mirrorstories.example") is None  # not a source
    cand = reg.status()["candidates"]
    assert cand and cand[0]["via"] == "book-page-link", cand
    desc = reg.candidates()[0]
    assert "book" in desc.kinds


def test_redirect_target_is_noted_untrusted(tmp_path):
    reg = _seeded()
    html = "<html><body>" + "<p>landing page prose</p>" * 30 + "</body></html>"
    fetcher = _real_fetcher({
        "https://freewebnovel.com/novel/alpha-tale":
            ("https://fwn-mirror.example/novel/alpha-tale", html)})
    adapter = FreeWebNovelAdapter(fetcher=fetcher)
    adapter.bind_book_registry(reg)
    fetcher.get("https://freewebnovel.com/novel/alpha-tale")
    cand_domains = [c.domain for c in reg.candidates()]
    assert cand_domains == ["fwn-mirror.example"], cand_domains
    meta = reg.status()["candidates"]
    assert meta[0]["via"] == "book-redirect", meta


def test_candidate_serves_nothing_until_promoted_then_grown_serves(tmp_path):
    reg = _seeded()
    assert reg.note_candidate(
        "https://freshink.example/novel/x", via="test",
        kinds=("book",)) is True
    # not promoted: no grown adapter, not in any serving list
    assert book_sources.grown_book_adapters(reg) == []
    assert reg.for_domain("freshink.example") is None
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] == 1, summary
    grown = book_sources.grown_book_adapters(reg)
    assert len(grown) == 1 and grown[0].domains == ("freshink.example",)
    # the promoted source actually serves through the universal
    # generic shape: chapter prose out of fixture markup
    body = " ".join(["Fresh ink prose carried the chapter forward."] * 12)
    grown[0].fetcher = _FakeFetcher({
        "https://freshink.example/ch1": f"""<html><body>
<div class="chapter-content"><p>{body}</p><p>{body}</p></div>
</body></html>"""})
    grown[0].bind_book_registry(reg)
    ch = grown[0].fetch_chapter("https://freshink.example/ch1")
    assert "Fresh ink" in ch.text
    st = reg.stats_for(reg.for_domain("freshink.example"))
    assert st["ok"] == 1, st
    # adapter_for_url with the registry routes a grown host to the
    # grown adapter, not the anonymous generic
    ad = book_sources.adapter_for_url(
        "https://freshink.example/novel/x", source_registry=reg)
    assert ad.name == "grown:freshink.example", ad.name


# ── (6) health loop + the existing scheduler job cover books ──────────


def test_health_loop_verifies_the_book_class():
    reg = _seeded()
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["checked"] >= len(reg)
    books = reg.status()["books"]
    assert books["sources"], books
    assert all(r["health"] == "verified-live" for r in books["sources"]), \
        books["sources"]
    reg2 = _seeded()
    for _ in range(3):
        reg2.verify_health(fetcher=_dead_fetcher)
    row = [r for r in reg2.status()["books"]["sources"]
           if r["domain"] == "freewebnovel.com"][0]
    assert row["health"] in ("stood-down", "disabled", "dead"), row


def _sched_ctx(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.tools.registry import ToolRegistry

    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    return ctx, registry


def test_existing_health_job_covers_book_sources(tmp_path, monkeypatch):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.media import source_registry as sr
    from nomorals.media.source_jobs import ensure_source_health_jobs

    monkeypatch.setattr(sr, "_default_health_fetch", _ok_fetcher)
    ctx, _registry = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = mock.MagicMock()
    ensure_source_health_jobs(sched)
    row = [j for j in sched.list_jobs()
           if j["name"] == "download source health"][0]
    payload = json.loads(ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE name = ?",
        ("download source health",))["payload"] or "{}")
    assert payload == {"tool": "download_sources",
                       "args": {"action": "health"}}
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    from nomorals.media.downloader import MediaDownloader
    books = MediaDownloader(ctx).source_registry().status()["books"]
    assert books["sources"] and all(
        r["health"] == "verified-live" for r in books["sources"]), books


# ── (7) persistence + status surfaces ─────────────────────────────────


def test_book_outcomes_persist_across_restart(tmp_path):
    kv, _db = _kv(tmp_path)
    reg = _seeded(store=kv)
    adapter = FreeWebNovelAdapter(fetcher=_FakeFetcher({
        "https://freewebnovel.com/novel/alpha-tale": FWN_NOVEL}))
    adapter.bind_book_registry(reg)
    meta = adapter.novel("https://freewebnovel.com/novel/alpha-tale")
    assert meta.title == "Alpha Tale"
    assert reg.note_candidate("https://freshink.example/novel/x",
                              via="test", kinds=("book",)) is True
    from nomorals.media.source_registry import _PERSIST_KEY
    raw = kv.get(_PERSIST_KEY)
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["version"] == 2  # no version bump (#88 precedent)

    reg2 = _seeded(store=kv)  # restart
    st = reg2.stats_for(reg2.for_domain("freewebnovel.com"))
    assert st["ok"] == 1, st
    assert [c.domain for c in reg2.candidates()] == ["freshink.example"]
    books = reg2.status()["books"]
    row = [r for r in books["sources"] if r["domain"] == "freewebnovel.com"]
    assert row and row[0]["ok"] == 1 and row[0]["origin"] == "seed", row
    assert books["candidates_pending"] == 1


def test_status_surfaces_show_the_book_class(tmp_path):
    reg = _seeded()
    reg.note_candidate("https://freshink.example/novel/x", via="test",
                       kinds=("book",))
    st = reg.status()
    books = st["books"]
    assert books["candidates_pending"] == 1
    ids = {r["source"] for r in books["sources"]}
    assert {"freewebnovel", "novelfull", "royalroad", "gutenberg"} <= ids
    from nomorals.media.source_registry import render_status_text
    text = render_status_text(st)
    assert "book sources" in text.lower(), text
    assert "freewebnovel.com" in text, text
    # the shared download_sources tool carries the same class
    ctx, registry = _sched_ctx(tmp_path)
    outcome = registry.call("download_sources", action="status")
    assert outcome.ok, getattr(outcome, "error", outcome)
    payload = getattr(outcome, "value", None) or outcome.data
    assert "books" in payload and payload["books"]["sources"], payload


def test_story_sources_tool_and_novel_surface(tmp_path):
    ctx, registry = _sched_ctx(tmp_path)
    outcome = registry.call("story_sources", action="status")
    assert outcome.ok, getattr(outcome, "error", outcome)
    payload = getattr(outcome, "value", None) or outcome.data
    assert payload["sources"], payload
    assert any(r["source"] == "freewebnovel" for r in payload["sources"])

    from nomorals.agents.partner.runtime_search import RuntimeSearchMixin
    obj = RuntimeSearchMixin.__new__(RuntimeSearchMixin)
    obj.context = ctx
    text = obj._control_novel("sources", "")
    assert "book sources" in text.lower(), text
    assert "freewebnovel" in text, text
    # the plain usage path is untouched
    usage = obj._control_novel("", "")
    assert "follow" in usage, usage
