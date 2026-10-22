"""Pass #94 — wisdom archive sources go open (the owner's
open-sources law).

Passes #86–#93 took downloads, music, books, web search, prices,
images and film open.  The last named CLOSED-SOURCE class was the
WisdomKeeper archives.  Survey truth this suite pins:

* The class's organ is the archive hunt in ``wisdom/ingestor.py``:
  ``ArchiveIngestor.search`` tried ``SEARCH_SOURCES``
  (gutendex → wikisource → archive → web) in a FIXED order,
  kept a backend's failure only in a local list that was thrown
  away unless *every* backend failed, recorded NO outcome
  anywhere, had NO learned ordering, NO candidate intake and NO
  health coverage; the ingest fetch path (``fetch_bytes``)
  credited no host for anything either.  Two of the four backend
  hosts (gutendex.com, en.wikisource.org) were not registered
  anywhere.
* ARCHITECTURE DECISION, with evidence: the class joins the SAME
  media registry as its own class — no mirrored machinery.  Every
  backend is one host serving the hunt, so the host-keyed model
  fits; two hosts are ALREADY descriptors (``archive.org``, the
  #88 seed; ``lite.duckduckgo.com``, the #90 web-search seed —
  the web backend fetches that literal host) and both classes
  fetch the literal same hosts, so the wisdom class SHARES those
  descriptors and extends their kinds (the #88 youtube.com / #92
  Bing-Yandex precedent).  The new seeds declare NO extraction
  facts, so the #76/#87 extractor view is byte-identical (pinned
  below).
* Outcome taxonomy (the one vocabulary): results landed =
  served; a backend that answered with no hits = weak evidence
  (``record_no_result``, #88 — an obscure text must never stand
  an archive down); a bot-wall / refusal (HTTP 401/403, or a
  robots.txt denial — the host's own published wall) is
  ``record_blocked`` (#89 — health ``bot-blocked``, never
  ``dead``); an HTTP 429 is ``record_rate_limited`` (#90); a
  transport failure / dead route is a failure with an error
  class.  The taxonomy is readable because the backends RAISE
  with the HTTP layer's error chained — the seam classifies off
  the cause chain (``http_status_of``), unlike the film chain,
  whose sources flattened errors and had to grow fetch evidence.
* Learned ordering applies to the DEFAULT hunt only: an
  explicit ``sources=`` argument is the caller's instruction
  (autonomy's gap-hunt passes a subset) and is never reordered.
* Health loop: the ONE existing loop / job covers the class;
  the two new seeds carry their REAL search URL template and
  are probed through it with one bounded probe query
  ("meditations" — the agency registry's wisdom probe), because
  a root page proves nothing about search (#90 reasoning).
  archive.org / lite.duckduckgo.com are claimed by the film /
  web probe routes (one probe per descriptor per loop) and are
  judged for wisdom by real wisdom-search outcomes.
* Discovery is genuine on two real flows: foreign hosts among
  search results (the web backend returns archive pages —
  sacred-texts and its kin) intake as untrusted candidates with
  ``kinds=("wisdom-archive",)``; and archive hosts the ingest
  fetch path successfully serves from (the corpus manifest and
  the research-fed queue bring new archives into the corpus)
  intake the same way.  A promoted archive serves through the
  generic ingest fetch path itself (``fetch_bytes`` is
  host-agnostic) — there is NO generic archive-search shape
  (nothing like the film chain's WordPress shape), so grown
  sources do not join the hunt's backend list: named absence,
  not invention.  An unpromoted candidate serves nothing.

All fetches are stubbed — no network.
"""

from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.core.errors import ProviderError
from nomorals.core.http import HttpResponse
from nomorals.wisdom import ArchiveIngestor, IngestError
from nomorals.wisdom import ingestor as ing_mod


# ── helpers ───────────────────────────────────────────────────────────


def _kv():
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


class _FakeHttp:
    """Stand-in for HttpClient: handler(url) -> HttpResponse or raises."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        return self.handler(url)


def _resp(body: bytes, url: str = "") -> HttpResponse:
    return HttpResponse(status=200, body=body, headers={}, url=url)


def _ctx(tmp: str, db=None):
    ns = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))
    if db is not None:
        ns.db = db
    return ns


def _make_ingestor(handler=None, registry=None):
    """An ArchiveIngestor with stubbed HTTP + permissive robots.

    The context carries NO database, so only an injected registry
    activates the seam — the db-less default stays inert.
    """
    tmp = tempfile.mkdtemp(prefix="pass94-wisdom-")
    ing = ArchiveIngestor(_ctx(tmp), registry=registry)
    ing._http = _FakeHttp(handler or (lambda url: _resp(b"", url=url)))
    ing._robots = SimpleNamespace(allowed=lambda *a, **k: True)
    return ing


def _gutendex_json(books=((2680, "Meditations"),)):
    payload = {"results": [
        {"id": gid, "title": title,
         "authors": [{"name": "A. Uthor"}],
         "subjects": ["Philosophy", "Ethics"]}
        for gid, title in books]}
    return json.dumps(payload).encode()


def _wikisource_json(titles=("Meditations",)):
    payload = {"query": {"search": [
        {"title": t, "snippet": "a snippet"} for t in titles]}}
    return json.dumps(payload).encode()


def _archive_json(idents=("meditations",)):
    payload = {"response": {"docs": [
        {"identifier": i, "title": i.title(), "description": ""}
        for i in idents]}}
    return json.dumps(payload).encode()


_DDG_HTML = (
    '<html><body>'
    '<a class="result-link" href="https://sacred-texts.com/bib/thomas.htm">'
    'Gospel of Thomas at sacred-texts</a>'
    '<a class="result-link" href="https://www.gutenberg.org/cache/epub/'
    '2680/pg2680.txt">Meditations on Gutenberg</a>'
    '</body></html>')


def _all_ok(url):
    if "gutendex.com" in url:
        return _resp(_gutendex_json(), url=url)
    if "en.wikisource.org" in url:
        return _resp(_wikisource_json(), url=url)
    if "archive.org/advancedsearch" in url:
        return _resp(_archive_json(), url=url)
    if "lite.duckduckgo.com" in url:
        return _resp(_DDG_HTML.encode(), url=url)
    raise AssertionError(f"unexpected URL {url}")


def _http_error(status: int, url: str = "https://x/"):
    return ProviderError(
        f"{status} from {url}", details={"http_status": status})


def _desc(reg, name):
    return reg.for_domain(ing_mod.WISDOM_SOURCE_DOMAINS[name])


# ── (1) seeds & membership ────────────────────────────────────────────


def test_every_backend_is_a_registered_seed():
    reg = _seeded()
    for name, domain in ing_mod.WISDOM_SOURCE_DOMAINS.items():
        desc = reg.for_domain(domain)
        assert desc is not None, f"{name} ({domain}) not registered"
        assert reg.origin_for(desc) == "seed"
        assert "wisdom-archive" in desc.kinds


def test_membership_map_matches_search_sources():
    assert tuple(ing_mod.WISDOM_SOURCE_DOMAINS) == \
        ArchiveIngestor.SEARCH_SOURCES


def test_shared_hosts_carry_both_class_memberships():
    reg = _seeded()
    archive = reg.for_domain("archive.org")
    assert "music-search" in archive.kinds   # the #88 class, untouched
    assert "film-search" in archive.kinds    # the #93 class, untouched
    assert "wisdom-archive" in archive.kinds  # extended, one descriptor
    ddg = reg.for_domain("lite.duckduckgo.com")
    assert "web-search" in ddg.kinds         # the #90 class, untouched
    assert "wisdom-archive" in ddg.kinds     # extended, one descriptor
    # a host the hunt never fetches gains no wisdom membership
    assert "wisdom-archive" not in reg.for_domain("bing.com").kinds


def test_new_seeds_carry_real_search_templates():
    reg = _seeded()
    gd = reg.for_domain("gutendex.com")
    assert gd.search_url_template == \
        "https://gutendex.com/books/?search={query}"
    ws = reg.for_domain("en.wikisource.org")
    assert ws.search_url_template.startswith(
        "https://en.wikisource.org/w/api.php?action=query")
    assert "{query}" in ws.search_url_template


def test_extractor_view_unchanged_by_wisdom_seeds():
    from nomorals.media.download_kit import descriptors_for_registry

    reg = _seeded()
    for domain in ("gutendex.com", "en.wikisource.org"):
        desc = reg.for_domain(domain)
        assert desc.extraction_rule == "" and desc.url_patterns == ()
    descs = descriptors_for_registry(reg)
    # the pristine tree derives exactly 13 extractor descriptors;
    # the wisdom seeds add none and change none
    assert len(descs) == 13


def test_status_carries_the_wisdom_class():
    reg = _seeded()
    wisdom = reg.status()["wisdom"]
    rows = {r["source"]: r for r in wisdom["sources"]}
    assert set(rows) == set(ArchiveIngestor.SEARCH_SOURCES)
    for name, row in rows.items():
        assert row["domain"] == ing_mod.WISDOM_SOURCE_DOMAINS[name]
        assert row["origin"] == "seed"
        assert row["ok"] == row["fail"] == 0
    assert wisdom["grown"] == [] and wisdom["candidates_pending"] == 0


def test_render_status_text_shows_wisdom_rows():
    from nomorals.media.source_registry import render_status_text

    text = render_status_text(_seeded().status())
    assert "📚 wisdom archive sources" in text
    assert "📚 gutendex → gutendex.com" in text
    assert "📚 web → lite.duckduckgo.com" in text


# ── (2) outcome taxonomy at the search seam ───────────────────────────


def test_served_search_credits_the_backend():
    reg = _seeded()
    ing = _make_ingestor(_all_ok, registry=reg)
    results = ing.search("meditations")
    assert results  # the hunt still returns candidates
    st = reg.stats_for(_desc(reg, "gutendex"))
    assert st["ok"] == 1 and st["fail"] == 0


def test_no_result_is_weak_evidence_not_failure():
    reg = _seeded()

    def handler(url):
        if "gutendex.com" in url:
            return _resp(_gutendex_json(books=()), url=url)
        return _all_ok(url)

    ing = _make_ingestor(handler, registry=reg)
    with pytest.raises(IngestError):
        # a lone backend answering empty still ends in the
        # pre-#94 all-empty raise — but the outcome is recorded
        # before the raise, in the weak-evidence class
        ing.search("some obscure text", sources=("gutendex",))
    st = reg.stats_for(_desc(reg, "gutendex"))
    assert st["no_result"] == 1 and st["fail"] == 0 and st["ok"] == 0
    # weak evidence never stands a source down
    assert reg.is_active(_desc(reg, "gutendex"))


def test_http_403_records_blocked_not_dead():
    reg = _seeded()

    def handler(url):
        if "gutendex.com" in url:
            raise _http_error(403, url)
        return _all_ok(url)

    ing = _make_ingestor(handler, registry=reg)
    ing.search("meditations", sources=("gutendex", "archive"))
    desc = _desc(reg, "gutendex")
    st = reg.stats_for(desc)
    assert st["blocked"] == 1 and st["last_error_class"] == "bot_block"
    assert reg.health_for(desc) == "bot-blocked"


def test_http_429_records_rate_limited():
    reg = _seeded()

    def handler(url):
        if "en.wikisource.org" in url:
            raise _http_error(429, url)
        return _all_ok(url)

    ing = _make_ingestor(handler, registry=reg)
    ing.search("meditations", sources=("wikisource", "archive"))
    desc = _desc(reg, "wikisource")
    st = reg.stats_for(desc)
    assert st["rate_limited"] == 1
    assert reg.health_for(desc) == "rate-limited"


def test_robots_denial_records_blocked():
    reg = _seeded()
    ing = _make_ingestor(_all_ok, registry=reg)
    ing._robots = SimpleNamespace(allowed=lambda *a, **k: False)
    with pytest.raises(IngestError):
        ing.search("meditations", sources=("archive",))
    st = reg.stats_for(_desc(reg, "archive"))
    assert st["blocked"] == 1 and st["fail"] == 1


def test_transport_failure_records_failure_with_error_class():
    reg = _seeded()

    def handler(url):
        if "gutendex.com" in url:
            raise ProviderError("connection refused", retryable=True)
        return _all_ok(url)

    ing = _make_ingestor(handler, registry=reg)
    ing.search("meditations", sources=("gutendex", "archive"))
    st = reg.stats_for(_desc(reg, "gutendex"))
    assert st["fail"] == 1 and st["ok"] == 0
    assert st["last_error_class"] == "ProviderError"


def test_http_500_records_http_failure_class():
    reg = _seeded()

    def handler(url):
        if "archive.org/advancedsearch" in url:
            raise _http_error(500, url)
        return _all_ok(url)

    ing = _make_ingestor(handler, registry=reg)
    ing.search("meditations", sources=("archive", "gutendex"))
    st = reg.stats_for(_desc(reg, "archive"))
    assert st["fail"] == 1 and st["last_error_class"] == "http_500"


def test_all_backends_down_still_raises_and_records_each():
    reg = _seeded()
    ing = _make_ingestor(
        lambda url: (_ for _ in ()).throw(
            ProviderError("everything down", retryable=True)),
        registry=reg)
    with mock.patch("time.sleep"):
        with pytest.raises(IngestError):
            ing.search("thomas")
    for name in ArchiveIngestor.SEARCH_SOURCES:
        assert reg.stats_for(_desc(reg, name))["fail"] == 1


def test_search_results_intake_foreign_hosts_as_wisdom_candidates():
    reg = _seeded()
    ing = _make_ingestor(_all_ok, registry=reg)
    ing.search("thomas", sources=("web",))
    cand_domains = {d.domain for d in reg.candidates()}
    assert "sacred-texts.com" in cand_domains
    cand = next(d for d in reg.candidates()
                if d.domain == "sacred-texts.com")
    assert "wisdom-archive" in cand.kinds
    # a host the registry already knows refuses candidate intake
    assert "gutenberg.org" not in cand_domains
    assert reg.wisdom_status()["candidates_pending"] >= 1


# ── (3) learned ordering ──────────────────────────────────────────────


def test_day_one_hunt_order_is_untouched():
    reg = _seeded()
    calls: list[str] = []

    def handler(url):
        calls.append(url)
        return _all_ok(url)

    ing = _make_ingestor(handler, registry=reg)
    ing.search("meditations", max_results=1)
    # gutendex is first in SEARCH_SOURCES and fills the quota —
    # nothing else is even attempted on day one
    assert calls and "gutendex.com" in calls[0]
    assert not any("wikisource" in c for c in calls)


def test_early_break_leaves_later_backends_unevidenced():
    reg = _seeded()
    ing = _make_ingestor(_all_ok, registry=reg)
    ing.search("meditations", max_results=2)
    # gutendex (1) + wikisource (1) fill max_results=2; archive and
    # web are never attempted and record nothing
    assert reg.stats_for(_desc(reg, "gutendex"))["ok"] == 1
    assert reg.stats_for(_desc(reg, "wikisource"))["ok"] == 1
    assert reg.stats_for(_desc(reg, "archive"))["ok"] == 0
    assert reg.stats_for(_desc(reg, "web"))["ok"] == 0


def test_learned_order_demotes_stood_down_backend():
    reg = _seeded()
    gd = _desc(reg, "gutendex")
    # stand gutendex down with consecutive blocked outcomes
    for _ in range(10):
        reg.record_blocked(gd)
    assert not reg.is_active(gd)
    calls: list[str] = []

    def handler(url):
        calls.append(url)
        return _all_ok(url)

    ing = _make_ingestor(handler, registry=reg)
    ing.search("meditations", max_results=1)
    # wikisource now leads the default hunt; gutendex is demoted
    assert "en.wikisource.org" in calls[0]


def test_explicit_sources_order_is_never_reordered():
    reg = _seeded()
    gd = _desc(reg, "gutendex")
    for _ in range(10):
        reg.record_blocked(gd)
    assert not reg.is_active(gd)
    calls: list[str] = []

    def handler(url):
        calls.append(url)
        return _all_ok(url)

    ing = _make_ingestor(handler, registry=reg)
    # the caller's explicit subset + order (autonomy's gap-hunt
    # shape) is an instruction — gutendex still goes first
    ing.search("meditations", sources=("gutendex", "archive"),
               max_results=1)
    assert "gutendex.com" in calls[0]


# ── (4) the ingest fetch seam ─────────────────────────────────────────

_TEXT = b"# Meditations\n\n" + b"Marcus Aurelius on fate. " * 40
_TEXT_URL = "https://sacred-texts.com/bib/thomas.htm"


def test_fetch_serves_credit_known_host_and_intake_unknown_host():
    reg = _seeded()
    ing = _make_ingestor(lambda url: _resp(_TEXT, url=url), registry=reg)
    with mock.patch("time.sleep"):
        # known host (gutenberg.org is a registry seed): credited
        data = ing.fetch_bytes(
            "https://www.gutenberg.org/cache/epub/2680/pg2680.txt")
        assert data == _TEXT
        st = reg.stats_for(reg.for_domain("gutenberg.org"))
        assert st["ok"] == 1
        # unknown host: the serve intakes it as a candidate
        ing.fetch_bytes(_TEXT_URL)
    cand_domains = {d.domain for d in reg.candidates()}
    assert "sacred-texts.com" in cand_domains


def test_fetch_cache_hit_earns_nothing():
    reg = _seeded()
    ing = _make_ingestor(lambda url: _resp(_TEXT, url=url), registry=reg)
    with mock.patch("time.sleep"):
        ing.fetch_bytes(_TEXT_URL)
        assert len(reg.candidates()) == 1
        ing.fetch_bytes(_TEXT_URL)  # cache hit — no second intake
    assert len(reg.candidates()) == 1
    assert reg.candidates()[0] is not None
    meta = reg._candidate_meta[reg.candidates()[0].key]
    assert meta["seen"] == 1  # the cache hit never reached the seam


def test_fetch_failure_of_known_host_records_failure():
    reg = _seeded()

    def handler(url):
        raise ProviderError("connection refused", retryable=True)

    ing = _make_ingestor(handler, registry=reg)
    with mock.patch("time.sleep"):
        with pytest.raises(IngestError):
            ing.fetch_bytes(
                "https://www.gutenberg.org/cache/epub/2680/pg2680.txt")
    st = reg.stats_for(reg.for_domain("gutenberg.org"))
    assert st["fail"] == 1 and st["ok"] == 0


def test_fetch_failure_of_unknown_host_records_nothing():
    reg = _seeded()

    def handler(url):
        raise ProviderError("connection refused", retryable=True)

    ing = _make_ingestor(handler, registry=reg)
    with mock.patch("time.sleep"):
        with pytest.raises(IngestError):
            ing.fetch_bytes(_TEXT_URL)
    # no descriptor to blame, and a failure is not discovery
    assert reg.candidates() == []


def test_fetch_robots_denial_records_blocked_for_known_host():
    reg = _seeded()
    ing = _make_ingestor(lambda url: _resp(_TEXT, url=url), registry=reg)
    ing._robots = SimpleNamespace(allowed=lambda *a, **k: False)
    with pytest.raises(IngestError):
        ing.fetch_bytes(
            "https://www.gutenberg.org/cache/epub/2680/pg2680.txt")
    st = reg.stats_for(reg.for_domain("gutenberg.org"))
    assert st["blocked"] == 1


# ── (5) growth: promotion, grown serving, unpromoted silence ──────────


def test_candidate_promotes_and_grown_archive_serves_via_fetch():
    reg = _seeded()
    ing = _make_ingestor(lambda url: _resp(_TEXT, url=url), registry=reg)
    with mock.patch("time.sleep"):
        ing.fetch_bytes(_TEXT_URL)  # intakes sacred-texts.com
    assert reg.candidates()
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] >= 1
    assert reg.candidates() == []
    grown = reg.for_domain("sacred-texts.com")
    assert grown is not None and reg.origin_for(grown) == "grown"
    assert "sacred-texts.com" in {
        g["domain"] for g in reg.wisdom_status()["grown"]}
    # the grown archive now SERVES through the generic ingest
    # fetch path, and its serves credit its own descriptor
    with mock.patch("time.sleep"):
        ing2 = _make_ingestor(lambda url: _resp(_TEXT, url=url),
                              registry=reg)
        ing2.fetch_bytes("https://sacred-texts.com/other.htm")
    assert reg.stats_for(grown)["ok"] >= 1


def test_unpromoted_candidate_serves_nothing():
    reg = _seeded()
    assert reg.note_candidate(
        "https://newarchive.example/texts/x.htm",
        via="wisdom-search-result", kinds=("wisdom-archive",))
    # a candidate is not a source: lookups refuse it, the hunt's
    # backend list is unchanged
    assert reg.for_domain("newarchive.example") is None
    assert reg.for_url("https://newarchive.example/texts/x.htm") is None
    assert tuple(ing_mod.WISDOM_SOURCE_DOMAINS) == \
        ArchiveIngestor.SEARCH_SOURCES


# ── (6) health loop probes ────────────────────────────────────────────


def test_health_loop_probes_wisdom_search_endpoints():
    reg = _seeded()
    seen = []

    def fetcher(url):
        seen.append(url)
        return 200, b"<html><body>real content</body></html>"

    reg.verify_health(fetcher=fetcher)
    assert any("gutendex.com/books/?search=meditations" in u
               for u in seen)
    assert any("en.wikisource.org/w/api.php" in u
               and "srsearch=meditations" in u for u in seen)


def test_health_loop_wisdom_429_is_rate_limited_not_dead():
    reg = _seeded()

    def fetcher(url):
        if "gutendex" in url:
            return 429, b""
        return 200, b"<html><body>real content</body></html>"

    reg.verify_health(fetcher=fetcher)
    assert reg.health_for(reg.for_domain("gutendex.com")) == \
        "rate-limited"


def test_health_loop_wisdom_403_is_bot_blocked():
    reg = _seeded()

    def fetcher(url):
        if "wikisource" in url:
            return 403, b""
        return 200, b"<html><body>real content</body></html>"

    reg.verify_health(fetcher=fetcher)
    assert reg.health_for(
        reg.for_domain("en.wikisource.org")) == "bot-blocked"


# ── (7) no-registry discipline + resolution ───────────────────────────


def test_no_registry_keeps_pre_pass_behaviour():
    ing = _make_ingestor(_all_ok)  # no registry injected, no db
    assert ing._registry is None
    results = ing.search("meditations")
    assert results  # the hunt works exactly as before
    # and with an inert DEFAULT registry nothing was recorded
    # anywhere the hunt could have reached (there is no seam)
    reg = _seeded()
    for name in ArchiveIngestor.SEARCH_SOURCES:
        assert reg.stats_for(_desc(reg, name))["ok"] == 0


def test_registry_resolution_discipline():
    assert ing_mod.wisdom_registry_for_context(None) is None
    assert ing_mod.wisdom_registry_for_context(
        SimpleNamespace(settings=None)) is None
    reg = _seeded()
    assert ing_mod.wisdom_registry_for_context(
        SimpleNamespace(db=object()), reg) is reg
    store, db = _kv()
    ctx = SimpleNamespace(db=db)
    bound = ing_mod.wisdom_registry_for_context(ctx)
    from nomorals.media.source_registry import DEFAULT_REGISTRY
    assert bound is DEFAULT_REGISTRY


def test_kv_persistence_round_trip(tmp_path):
    store, db = _kv()
    reg = _seeded(store=store)
    ing = _make_ingestor(_all_ok, registry=reg)
    ing.search("meditations", sources=("gutendex",))
    assert reg.stats_for(_desc(reg, "gutendex"))["ok"] == 1
    # a fresh registry over the same store restores the outcome
    reg2 = _seeded(store=store)
    assert reg2.stats_for(_desc(reg2, "gutendex"))["ok"] == 1


# ── (8) the wisdom_sources tool surface ───────────────────────────────


def test_wisdom_sources_tool_registered_and_answers():
    from types import SimpleNamespace as NS

    from nomorals.tools import wisdom as wisdom_tools

    class _Reg:
        def __init__(self):
            self.context = NS(settings=NS(workspace_dir="/tmp"))
            self.tools = {}

        def register(self, name, **kw):
            def deco(fn):
                self.tools[name] = fn
                return fn
            return deco

    reg = _Reg()
    wisdom_tools.register(reg)
    assert "wisdom_sources" in reg.tools
    out = reg.tools["wisdom_sources"](action="status")
    rows = {r["source"]: r for r in out["sources"]}
    assert set(rows) == set(ArchiveIngestor.SEARCH_SOURCES)
    assert reg.tools["wisdom_sources"](
        action="bogus")["error"].startswith("unknown action")
