"""Pass #92 — image sources go open (the owner's open-sources law).

Passes #86–#90 took downloads, music, books and web search open
through the ONE registry (``media/source_registry.py``); #91 opened
prices with mirrored machinery where layering law forbade the
import.  The remaining named CLOSED-SOURCE class was images.
Survey truth this suite pins:

* The ONLY image-search organ in the repo is reverse image search
  in ``tools/imagedb.py`` (the ``/lens`` surface, the partner
  runtime, role specs): a fixed roster of scraped engines —
  Bing visual search (``www.bing.com/images/search?...iss=sbi``)
  and Yandex (``yandex.com/images/search?rpt=imageview``) — run in
  the hardcoded order ``("bing", "yandex")`` by ``_scrape_engines``,
  each isolated, with NO outcome recording, NO health, NO
  persistence: results landed / zero results / dead engine /
  bot-walled engine were indistinguishable (every failure became
  ``[]``), so a dead or walled engine was retried in the same
  order forever.  Google Lens (``lens.google.com``) and TinEye
  (``tineye.com``) appear only as deep LINKS the tool hands the
  user — Devon never fetches them, so no outcome can ever be
  recorded for them honestly.
* There is NO forward (text→image) image-search organ, NO
  stock-photo/asset source (no Pexels/Unsplash/Pixabay image
  search anywhere in code — the Pixabay row in
  ``music_sources`` is a music-sample source), and the
  Openverse/Wikimedia sources in ``media/sources.py`` serve AUDIO
  only (``/v1/audio/``, ``filetype:audio``) — they are music-class
  members already (#88), not an image roster hiding elsewhere.
  ``/image`` with a path is LOCAL analysis (no remote source);
  ``/look`` is vision-model analysis (the LLM provider class, not
  image sources); image GENERATION (``media_edit/generate.py``
  backends, ``media/imggen``) selects ONE backend by explicit
  configuration — it does not share the search roster, has no
  fallback chain, and is a different class, out of scope here.
* ARCHITECTURE DECISION, with evidence: the image class joins the
  SAME media registry as its own class — no mirrored machinery.
  (1) Layering: ``imagedb`` is tools (L4) and ``media`` is L4;
  its peer ``tools/web.py`` already resolves the same registry
  the same lazy way (#90), and no module pin forbids it (the #91
  blocker was ``integrations/market_data`` pinned L2
  imports-only-core; imagedb has no such pin).  (2) Identity:
  every scraped reverse engine is one host serving image search,
  so the registry's host-keyed model fits without distortion —
  and the two hosts (``bing.com``, ``yandex.com``) are ALREADY
  registry descriptors as web-search seeds (#90): both classes
  fetch the literal same hosts (``www.bing.com/search`` vs
  ``www.bing.com/images/search``; ``yandex.com/search`` vs
  ``yandex.com/images/search``), so the image class shares those
  host descriptors and extends their kinds — exactly how
  ``youtube.com`` carries both download and music-search
  membership (#88).  The #90 Brave split into two descriptors
  was because the hosts genuinely differ
  (``search.brave.com`` vs ``api.search.brave.com``); here they
  do not.  Google Lens / TinEye get NO descriptor (the #90 ddgs
  precedent: never fetched → no credit can be given honestly);
  they are surfaced as link-only rows in the class status.
* Outcome taxonomy (the one vocabulary): results landed =
  served; an empty result page = weak evidence
  (``record_no_result``, #88 — an obscure image must never stand
  an engine down); a transport failure records with an error
  class; a bot-wall (HTTP 403, or a 200 page carrying a
  captcha/Cloudflare challenge) is ``record_blocked`` (#89 —
  health ``bot-blocked``, never ``dead``); an HTTP 429 is
  ``record_rate_limited`` (#90 — recorded once, never retried
  around: this suite counts the attempts).  The real
  ``HttpClient`` raises on HTTP status, so the status is read
  from the exception (``core.errors.http_status_of``) as well as
  from duck-typed responses.
* Health loop + persistence: the ONE existing registry health
  loop / health job covers the class (the shared host descriptors
  are probed through their real search endpoint — one bounded
  query on the same host, #90); the reverse endpoint itself is
  judged by real reverse-search outcomes, because a synthetic
  reverse probe would need a hosted reference image and returns
  a JS app shell — probing it would be theater, stated plainly.
  Learned state persists through the same KV discipline,
  version unchanged at 2 (#88–#90 precedent: new stats counters
  need no bump — this class adds none).  Status shows on the
  EXISTING surfaces, extended: ``SourceRegistry.status()``
  carries the class under ``images``, the shared
  ``download_sources`` tool carries it too, and the image home
  gains an ``image_sources`` tool mirroring #90's
  ``search_sources``.
* Discovery verdict — NAMED ABSENCE, no invention: reverse-search
  RESULT hosts are image CONTENT (arbitrary CDNs/blogs named by
  ``mediaurl``), not engines — a host that hosts a matching
  picture cannot be queried BY picture, so it can never serve
  this class.  No genuine discovery channel exists in code, so
  the class ships outcomes / health / ordering / persistence
  with NO candidate intake — a reverse search must leave the
  candidate list untouched, pinned below.

All fetches are stubbed — no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.tools import imagedb


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


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


def _dead_fetcher(url):
    return 404, b"gone"


_BING_HTML = (
    b'<html><body><div class="iusc" m="{x}">'
    b'<a mediaurl="https://example.com/first.jpg" href="#">x</a>'
    b'<a mediaurl="https://example.org/second.png" href="#">y</a>'
    b"</div></body></html>"
)
_YANDEX_HTML = (
    b'<html><body><script>'
    b'{"originalImage":{"url":"https://avatars.yandex.net/img1.jpg"}}'
    b"</script>"
    b'<img class="serp-item__thumb" src="https://example.org/hit.png">'
    b"</body></html>"
)
_BOTWALL_HTML = b"<html><body>Please verify you are human - captcha</body></html>"


class _Resp:
    def __init__(self, body=b"", status=200):
        self.body = body
        self.status = status
        self.ok = 200 <= status < 300


class _ImageClient:
    """Fake reverse-search client routing by engine host fragment.

    ``behaviour`` maps an engine name to one of: ("results", html),
    ("empty",), ("raise", exc), ("http", status),
    ("botwall",), ("badparse",).  Unlisted engines answer an empty
    page (a no-result).
    """

    _HOSTS = {"bing.com": "bing", "yandex.com": "yandex"}

    def __init__(self, behaviour=None):
        self.behaviour = behaviour or {}
        self.calls = []

    def get(self, url):
        engine = next((name for frag, name in self._HOSTS.items()
                       if frag in url), "?")
        self.calls.append(engine)
        action = self.behaviour.get(engine, ("empty",))
        kind = action[0]
        if kind == "raise":
            raise action[1]
        if kind == "http":
            return _Resp(b"", status=action[1])
        if kind == "botwall":
            return _Resp(_BOTWALL_HTML, status=200)
        if kind == "badparse":
            return _Resp(None, status=200)
        if kind == "results":
            return _Resp(action[1], status=200)
        return _Resp(b"<html><body>no matches</body></html>", status=200)


_IMG = "https://img.test/photo.jpg"


# ── (1) the image class joins the SAME registry ───────────────────────


def test_every_scraped_engine_is_a_registered_seed():
    reg = _seeded()
    origins = {s["domain"]: s["origin"] for s in reg.status()["sources"]}
    for engine, domain in imagedb.IMAGE_ENGINE_DOMAINS.items():
        assert domain, engine
        desc = reg.for_domain(domain)
        assert desc is not None, f"{engine} ({domain}) not registered"
        assert "image-search" in desc.kinds, (engine, desc.kinds)
        assert "image" in desc.kinds, (engine, desc.kinds)
        assert origins[domain] == "seed", (engine, origins[domain])


def test_membership_maps_cover_the_whole_roster():
    assert set(imagedb.IMAGE_ENGINE_DOMAINS) == \
        set(imagedb.REVERSE_IMAGE_ENGINES)
    assert set(imagedb.REVERSE_IMAGE_ENGINES) == {"bing", "yandex"}
    # domains are the hosts the scrapers actually fetch (evidence
    # from _scrape_bing/_scrape_yandex URL construction, www-stripped
    # by the registry's own canonicalisation) — the same hosts the
    # web-search class fetches its search pages from (#90)
    assert imagedb.IMAGE_ENGINE_DOMAINS["bing"] == "bing.com"
    assert imagedb.IMAGE_ENGINE_DOMAINS["yandex"] == "yandex.com"
    from nomorals.tools.web import SEARCH_ENGINE_DOMAINS

    assert imagedb.IMAGE_ENGINE_DOMAINS["bing"] == \
        SEARCH_ENGINE_DOMAINS["bing"]
    assert imagedb.IMAGE_ENGINE_DOMAINS["yandex"] == \
        SEARCH_ENGINE_DOMAINS["yandex"]


def test_link_only_engines_have_no_descriptor_and_earn_no_credit():
    reg = _seeded()
    assert imagedb.IMAGE_LINK_ONLY_DOMAINS == {
        "google_lens": "lens.google.com", "tineye": "tineye.com"}
    for domain in imagedb.IMAGE_LINK_ONLY_DOMAINS.values():
        # never fetched by code → no descriptor, no invented credit
        # (the #90 ddgs precedent)
        assert reg.for_domain(domain) is None, domain


def test_image_seeds_do_not_change_the_extractor_view():
    from nomorals.media.download_kit import descriptors_for_registry

    names = [d.name for d in descriptors_for_registry(_seeded())]
    assert names == [
        "netnaija", "audiomack", "boomplay", "notjustok", "tooxclusive",
        "youtube", "soundcloud", "spotify", "tiktok", "instagram",
        "x-twitter", "direct-media", "ytdlp-generic"], names


def test_imagedb_is_not_layer_blocked_from_the_registry():
    # the #91 lesson: check the module's layer pin before assuming
    # the registry import is legal — imagedb (tools, L4) has no
    # pin forbidding the L4 media registry, and its peer tools/web
    # already imports it the same lazy way
    import ast
    from pathlib import Path

    tree = ast.parse(Path(imagedb.__file__).read_text(encoding="utf-8"))
    top_imports = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module)
        elif isinstance(node, ast.Import):
            top_imports.update(a.name for a in node.names)
    # no TOP-LEVEL media import (the registry link stays lazy, so
    # the registry's own lazy import of this module cannot cycle)
    assert not any("source_registry" in m for m in top_imports), \
        top_imports


# ── (2) day-one pin: no learned state → today's order + payload ───────


def test_day_one_engine_order_is_byte_for_byte():
    reg = _seeded()
    assert imagedb.ordered_image_engines(reg) == \
        list(imagedb.REVERSE_IMAGE_ENGINES) == ["bing", "yandex"]
    behaviour = {"bing": ("results", _BING_HTML),
                 "yandex": ("results", _YANDEX_HTML)}
    plain_client = _ImageClient(behaviour)
    plain = imagedb._scrape_engines(plain_client, _IMG)
    learned_client = _ImageClient(behaviour)
    learned = imagedb._scrape_engines(
        learned_client, _IMG, registry=reg)
    assert learned_client.calls == plain_client.calls == \
        ["bing", "yandex"]
    assert learned == plain


def test_day_one_reverse_payload_is_unchanged_by_a_registry(tmp_path):
    root = tmp_path / "home"
    root.mkdir()
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    class URLClient:
        def get(self, url):
            if "bing.com" in url:
                return SimpleNamespace(body=_BING_HTML)
            if "yandex.com" in url:
                return SimpleNamespace(body=_YANDEX_HTML)
            return SimpleNamespace(body=png)

    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.execute(
        "CREATE TABLE IF NOT EXISTS image_index ("
        "hash TEXT PRIMARY KEY, path TEXT NOT NULL DEFAULT '', "
        "size INTEGER NOT NULL DEFAULT 0, mime TEXT NOT NULL DEFAULT '', "
        "first_seen REAL NOT NULL DEFAULT 0, last_seen REAL NOT NULL "
        "DEFAULT 0, seen_in TEXT NOT NULL DEFAULT '[]')")
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(home=str(root), tools=None))
    reg = _seeded()
    with mock.patch("nomorals.tools.imagedb.HttpClient",
                    return_value=URLClient()):
        plain = imagedb._reverse_impl(ctx, _IMG)
    with mock.patch("nomorals.tools.imagedb.HttpClient",
                    return_value=URLClient()):
        learned = imagedb._reverse_impl(ctx, _IMG, registry=reg)
    assert learned == plain
    assert learned["unverified_by_engine"]["bing"]
    assert learned["unverified_by_engine"]["yandex"]
    assert "google_lens" in learned["lookup_links"]


# ── (3) outcome recording on the real scrape paths ────────────────────


def test_chain_outcomes_are_recorded_per_engine():
    reg = _seeded()
    client = _ImageClient({
        "bing": ("results", _BING_HTML),
        "yandex": ("raise", RuntimeError("yandex down")),
    })
    out = imagedb._scrape_engines(client, _IMG, registry=reg)
    assert out["bing"] and out["yandex"] == []
    # served results credit the serving engine
    st = reg.stats_for(reg.for_domain("bing.com"))
    assert st["ok"] == 1 and st["fail"] == 0, st
    # a transport failure records with an error class
    st = reg.stats_for(reg.for_domain("yandex.com"))
    assert st["fail"] == 1 and st["last_error_class"] == \
        "network_failure", st
    # exactly one attempt per engine: a failure is never retried around
    assert client.calls == ["bing", "yandex"], client.calls


def test_empty_page_is_weak_evidence_never_a_failure():
    reg = _seeded()
    client = _ImageClient({})  # both engines answer an empty page
    out = imagedb._scrape_engines(client, _IMG, registry=reg)
    assert out == {"bing": [], "yandex": []}
    for domain in ("bing.com", "yandex.com"):
        desc = reg.for_domain(domain)
        st = reg.stats_for(desc)
        assert st["no_result"] == 1 and st["fail"] == 0, st
        assert reg.health_for(desc) == "unknown"
    # even repeated no-results never stand an engine down: an
    # obscure image says nothing about the engine serving others
    for _ in range(5):
        imagedb._scrape_engines(_ImageClient({}), _IMG, registry=reg)
    for domain in ("bing.com", "yandex.com"):
        assert reg.health_for(reg.for_domain(domain)) == "unknown"


def test_parser_failure_is_a_malformed_response():
    reg = _seeded()
    client = _ImageClient({"bing": ("badparse",)})
    imagedb._scrape_engines(client, _IMG, registry=reg)
    st = reg.stats_for(reg.for_domain("bing.com"))
    assert st["fail"] == 1 and st["last_error_class"] == \
        "malformed_response", st


def test_botwall_page_is_blocked_never_dead():
    reg = _seeded()
    client = _ImageClient({"bing": ("botwall",),
                           "yandex": ("http", 403)})
    imagedb._scrape_engines(client, _IMG, registry=reg)
    for domain in ("bing.com", "yandex.com"):
        desc = reg.for_domain(domain)
        st = reg.stats_for(desc)
        assert st["blocked"] == 1 and st["fail"] == 1, (domain, st)
        assert st["last_error_class"] == "bot_block", st
        assert reg.health_for(desc) == "bot-blocked", domain  # not dead


def test_rate_limited_is_its_own_class_and_a_single_attempt():
    reg = _seeded()
    client = _ImageClient({"bing": ("http", 429)})
    imagedb._scrape_engines(client, _IMG, registry=reg)
    assert client.calls.count("bing") == 1  # the 429 hard stop: no retry
    desc = reg.for_domain("bing.com")
    st = reg.stats_for(desc)
    assert st["rate_limited"] == 1 and st["fail"] == 1, st
    assert st["last_error_class"] == "rate_limited", st
    assert reg.health_for(desc) == "rate-limited"  # NOT dead


def test_http_status_on_a_raised_exception_is_classified():
    # the real HttpClient RAISES on HTTP status (http_error carries
    # details["http_status"]) — the scrape path must classify the
    # exception, not flatten it into a generic failure
    from nomorals.core.http import http_error

    reg = _seeded()
    client = _ImageClient({
        "bing": ("raise", http_error(429, "slow down", "https://x")),
        "yandex": ("raise", http_error(403, "forbidden", "https://x")),
    })
    imagedb._scrape_engines(client, _IMG, registry=reg)
    st = reg.stats_for(reg.for_domain("bing.com"))
    assert st["rate_limited"] == 1, st
    assert reg.health_for(reg.for_domain("bing.com")) == "rate-limited"
    st = reg.stats_for(reg.for_domain("yandex.com"))
    assert st["blocked"] == 1, st
    assert reg.health_for(reg.for_domain("yandex.com")) == "bot-blocked"


def test_search_results_do_not_intake_candidates():
    # Discovery verdict: result hosts are image CONTENT, not
    # engines — a real reverse search leaves candidates untouched.
    reg = _seeded()
    client = _ImageClient({"bing": ("results", _BING_HTML),
                           "yandex": ("results", _YANDEX_HTML)})
    imagedb._scrape_engines(client, _IMG, registry=reg)
    assert reg.candidates() == []
    assert reg.status()["images"]["candidates_pending"] == 0


# ── (4) ordering becomes outcome-aware ────────────────────────────────


def test_stood_down_engine_is_demoted_to_last():
    reg = _seeded()
    desc = reg.for_domain("bing.com")
    for _ in range(3):
        reg.record_outcome(desc, False, error_class="network_failure")
    assert reg.health_for(desc) == "stood-down"
    order = imagedb.ordered_image_engines(reg)
    assert order == ["yandex", "bing"], order  # demoted, still present
    client = _ImageClient({"yandex": ("results", _YANDEX_HTML)})
    imagedb._scrape_engines(client, _IMG, registry=reg)
    assert client.calls == ["yandex", "bing"], client.calls


def test_serving_engine_is_preferred_with_evidence():
    reg = _seeded()
    desc = reg.for_domain("yandex.com")
    for _ in range(3):
        reg.record_outcome(desc, True)
    order = imagedb.ordered_image_engines(reg)
    assert order[0] == "yandex", order


# ── (5) health loop + the existing scheduler job cover images ─────────


def test_health_loop_verifies_the_image_hosts():
    reg = _seeded()
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["checked"] >= len(reg)
    images = reg.status()["images"]
    assert images["engines"]
    assert all(r["health"] == "verified-live"
               for r in images["engines"]), images["engines"]
    # the dead-fetcher loop stands the class down by the same
    # vocabulary (3 consecutive failures → stood-down)
    reg2 = _seeded()
    for _ in range(3):
        reg2.verify_health(fetcher=_dead_fetcher)
    row = [r for r in reg2.status()["images"]["engines"]
           if r["domain"] == "bing.com"][0]
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


def test_existing_health_job_covers_images(tmp_path, monkeypatch):
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

    images = MediaDownloader(ctx).source_registry().status()["images"]
    assert images["engines"] and all(
        r["health"] == "verified-live" for r in images["engines"]), images


# ── (6) persistence + status surfaces ─────────────────────────────────


def test_image_outcomes_persist_across_restart(tmp_path):
    kv, _db = _kv(tmp_path)
    reg = _seeded(store=kv)
    client = _ImageClient({"bing": ("results", _BING_HTML),
                           "yandex": ("http", 429)})
    imagedb._scrape_engines(client, _IMG, registry=reg)
    from nomorals.media.source_registry import _PERSIST_KEY

    raw = kv.get(_PERSIST_KEY)
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["version"] == 2  # no version bump (#88–#90 precedent)

    reg2 = _seeded(store=kv)  # restart
    st = reg2.stats_for(reg2.for_domain("bing.com"))
    assert st["ok"] == 1, st
    st = reg2.stats_for(reg2.for_domain("yandex.com"))
    assert st["rate_limited"] == 1, st
    images = reg2.status()["images"]
    row = [r for r in images["engines"] if r["domain"] == "bing.com"]
    assert row and row[0]["ok"] == 1 and row[0]["origin"] == "seed", row


def test_status_surfaces_show_the_image_class(tmp_path):
    reg = _seeded()
    st = reg.status()
    images = st["images"]
    engine_names = {r["engine"] for r in images["engines"]}
    assert engine_names == {"bing", "yandex"}
    link_names = {r["engine"] for r in images["link_only"]}
    assert link_names == {"google_lens", "tineye"}
    assert all(r["health"] == "link-only" for r in images["link_only"])
    assert images["discovery"]  # the named-absence verdict, surfaced
    from nomorals.media.source_registry import render_status_text

    text = render_status_text(st)
    assert "image" in text.lower(), text
    assert "bing.com" in text, text
    # the shared download_sources tool carries the same class
    ctx, registry = _sched_ctx(tmp_path)
    outcome = registry.call("download_sources", action="status")
    assert outcome.ok, getattr(outcome, "error", outcome)
    payload = getattr(outcome, "value", None) or outcome.data
    assert "images" in payload and payload["images"]["engines"], payload


def test_image_sources_tool_surface(tmp_path):
    _ctx, registry = _sched_ctx(tmp_path)
    outcome = registry.call("image_sources", action="status")
    assert outcome.ok, getattr(outcome, "error", outcome)
    payload = getattr(outcome, "value", None) or outcome.data
    assert payload["engines"], payload
    assert any(r["engine"] == "bing" for r in payload["engines"])
    assert any(r["engine"] == "tineye" for r in payload["link_only"])
