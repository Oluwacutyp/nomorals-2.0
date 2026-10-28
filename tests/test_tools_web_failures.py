"""Tail-20 pins: tools/web.py failure values + NOT-recording.

Every site here is an exclusion: each handler returns a documented
fallback/miss value (or a bookkeeping pass that never breaks search),
and none of them records into the tools failures ledger. Each pin drives
the handler into its except path, pins the exact fallback value, and
asserts the ledger snapshot is unchanged (tail-8 method). No handler
here calls record_failure, and none of these failures may surface as a
crash or a success claim.

No except handlers in this file (try/finally only for patch restore),
so the census scanner counts zero sites here.
"""

import sys
import time

from nomorals.tools import registry_failures as tools_fail
from nomorals.tools.web import (
    RobotsCache,
    _engine_desc,
    _html_to_markdown,
    _record_engine_outcome,
    _trafilatura_extract,
    ordered_search_engines,
    readability_extract,
    web_registry_for_context,
)


def _ledger():
    return list(tools_fail.recent_failures())


class _FakeResp:
    def __init__(self, ok, text):
        self.ok = ok
        self.text = text


class _FailClient:
    def __init__(self):
        self.calls = 0

    def get(self, url):
        self.calls += 1
        raise ConnectionError("boom")


class _TextClient:
    def __init__(self, text):
        self.text = text

    def get(self, url):
        return _FakeResp(True, self.text)


class _BrokenParser:
    def can_fetch(self, *a):
        raise ValueError("malformed")


class _FakeReg:
    def __init__(self, fail_bind):
        self.fail_bind = fail_bind
        self.bound = []

    def bind_store(self, store):
        if self.fail_bind:
            raise RuntimeError("bind boom")
        self.bound.append(store)


class _DummyStore:
    def __init__(self, db):
        self.db = db


class _RaisingDbCtx:
    @property
    def db(self):
        raise RuntimeError("db boom")


class _RaisingReg:
    def for_domain(self, domain):
        raise RuntimeError("registry boom")


class _StatsRaisingReg:
    def for_domain(self, domain):
        return "D"

    def stats_for(self, desc):
        raise RuntimeError("stats boom")


class _RecordRaisingReg:
    def __init__(self):
        self.calls = 0

    def for_domain(self, domain):
        return "DESC"

    def record_outcome(self, *a, **k):
        self.calls += 1
        raise RuntimeError("record boom")


_HTML = "<html><body><article><p>" + "Hello world. " * 20 + "</p></article></body></html>"


def test_robots_fetch_failure_is_allowed_and_cached():
    # Site 107: unreachable robots.txt means "no restrictions known"
    # (documented policy); the None parser is cached, not re-fetched.
    client = _FailClient()
    cache = RobotsCache()
    before = _ledger()
    assert cache.allowed("http://example.com/page", client=client) is True
    assert cache.allowed("http://example.com/other", client=client) is True
    assert client.calls == 1
    assert RobotsCache().allowed(
        "http://example.com/page",
        client=_TextClient("User-agent: *\nDisallow: /")) is False
    assert _ledger() == before


def test_robots_malformed_parser_does_not_block():
    # Site 114: a malformed robots.txt must not block the fetch.
    cache = RobotsCache()
    cache._cache["http://example.com"] = (time.time(), _BrokenParser())
    before = _ledger()
    assert cache.allowed("http://example.com/page") is True
    assert _ledger() == before


def test_trafilatura_missing_is_none_and_falls_back():
    # Site 342: trafilatura is an optional dependency; None is the
    # documented miss value and the caller falls back to the scorer.
    real = sys.modules.get("trafilatura")
    sys.modules["trafilatura"] = None  # forces ImportError on import
    before = _ledger()
    try:
        assert _trafilatura_extract(_HTML) is None
        result = readability_extract(_HTML)
    finally:
        if real is not None:
            sys.modules["trafilatura"] = real
        else:
            del sys.modules["trafilatura"]
    assert result["text"].strip() != ""
    assert result["engine"] != "trafilatura"
    assert _ledger() == before


def test_trafilatura_extract_failure_is_none_and_falls_back():
    # Site 378: an extraction failure falls back to the built-in scorer.
    import types

    fake = types.ModuleType("trafilatura")
    fake.extract = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    real = sys.modules.get("trafilatura")
    sys.modules["trafilatura"] = fake
    before = _ledger()
    try:
        assert _trafilatura_extract(_HTML) is None
        result = readability_extract(_HTML)
    finally:
        if real is not None:
            sys.modules["trafilatura"] = real
        else:
            del sys.modules["trafilatura"]
    assert result["text"].strip() != ""
    assert _ledger() == before


def test_markdownify_missing_uses_stdlib_fallback():
    # Site 463: markdownify is optional; the stdlib fallback below is
    # the documented path, not a swallowed failure.
    real = sys.modules.get("markdownify")
    sys.modules["markdownify"] = None  # forces ImportError on import
    before = _ledger()
    try:
        out = _html_to_markdown("<h1>Hi</h1><p>Body text</p>")
    finally:
        if real is not None:
            sys.modules["markdownify"] = real
        else:
            del sys.modules["markdownify"]
    assert "# Hi" in out
    assert "Body text" in out
    assert _ledger() == before


def test_registry_bind_failure_still_returns_registry():
    # Site 546: persistence is an enhancement; its absence never sinks
    # a search — the registry is returned unbound (tail-18 site 250
    # shape, same documented contract).
    import nomorals.media.source_registry as src_reg
    import nomorals.storage.kv as kv_mod

    fake_reg = _FakeReg(True)
    real_reg, real_kv = src_reg.DEFAULT_REGISTRY, kv_mod.KVStore
    src_reg.DEFAULT_REGISTRY, kv_mod.KVStore = fake_reg, _DummyStore
    before = _ledger()
    try:
        got = web_registry_for_context(type("C", (), {"db": object()})())
    finally:
        src_reg.DEFAULT_REGISTRY, kv_mod.KVStore = real_reg, real_kv
    assert got is fake_reg
    assert fake_reg.bound == []
    assert _ledger() == before


def test_registry_outer_failure_is_none():
    # Site 550: the documented miss value ("or None", "Never raises").
    before = _ledger()
    assert web_registry_for_context(_RaisingDbCtx()) is None
    assert web_registry_for_context(type("C", (), {"db": None})()) is None
    assert _ledger() == before


def test_engine_desc_failure_is_none():
    # Site 562: the documented miss value for one chain engine.
    before = _ledger()
    assert _engine_desc(_RaisingReg(), "duckduckgo") is None
    assert _engine_desc(None, "duckduckgo") is None
    assert _ledger() == before


def test_ordering_hiccup_returns_chain_as_is():
    # Site 606: "Never raises; a registry hiccup returns the chain
    # as-is" — input order, no ranked claim (tail-18 site 394 shape).
    pool = ("duckduckgo", "bing")
    before = _ledger()
    assert ordered_search_engines(_StatsRaisingReg(), pool) == list(pool)
    assert ordered_search_engines(None, pool) == list(pool)
    assert _ledger() == before


def test_record_outcome_failure_never_breaks_search():
    # Site 636: bookkeeping must never break a search — the record
    # attempt is made exactly once, its failure swallowed, and the
    # caller (the engine chain) is unaffected (tail-18 record shape).
    reg = _RecordRaisingReg()
    before = _ledger()
    assert _record_engine_outcome(reg, "duckduckgo", "failure") is None
    assert reg.calls == 1
    assert _ledger() == before
