"""Pass #121 pins: SourceRegistry-class failures stop vanishing.

Covers the 28 converted SourceRegistry engine sites, the 2 deliberate
probe exclusions, the 3 debug-honest sites, and the ledger contract.
Every converted site is driven through the real method (with a raising
store/helper where the failure lives behind an injected dependency).
"""
import ast
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media.source_registry import (
    LearnedRule, Licence, SourceDescriptor, SourceRegistry, TrustTier,
)
from nomorals.media import source_registry as sr_mod
from nomorals.media.source_registry_failures import (
    clear_failures, failure_count, format_failures, recent_failures,
    record_failure,
)


# ── helpers ────────────────────────────────────────────────────────────


def _reg():
    return SourceRegistry()


def _desc(domain="example.com", **kw):
    kw.setdefault("nation", "NG")
    kw.setdefault("language", "en")
    kw.setdefault("domain", domain)
    kw.setdefault("name", domain)
    kw.setdefault("kinds", ("music",))
    return SourceDescriptor(**kw)


def _sites():
    return [f.site for f in recent_failures(100)]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


class _BadKey:
    """Descriptor-like object whose .key raises (malformed descriptor)."""
    domain = "example.com"
    nation = "NG"

    @property
    def key(self):
        raise RuntimeError("boom-key")


class _RaisingStore:
    """KV store whose reads/writes all fail."""
    def get(self, key):
        raise RuntimeError("store-get-boom")

    def delete(self, key):
        raise RuntimeError("store-delete-boom")


@pytest.fixture(autouse=True)
def _clean():
    clear_failures()
    yield
    clear_failures()


# ── ledger contract ────────────────────────────────────────────────────


def test_ledger_basics_and_format():
    assert recent_failures() == []
    assert format_failures() == "no recorded source registry failures"
    record_failure("SourceRegistry.test", ValueError("bad value"))
    entries = recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "SourceRegistry.test"
    assert entries[0].exc_type == "ValueError"
    assert entries[0].message == "bad value"
    assert format_failures() == "SourceRegistry.test: ValueError: bad value"
    assert failure_count() == 1


def test_ledger_bound_and_unprintable():
    for _ in range(250):
        record_failure("SourceRegistry.test", RuntimeError("x"))
    assert len(recent_failures(1000)) == 200
    assert failure_count() == 250

    class _Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    clear_failures()
    record_failure("SourceRegistry.test", _Unprintable())
    assert "unprintable" in recent_failures()[0].message


def test_direct_import_surface_and_all_unchanged():
    import nomorals.media.source_registry_failures as led
    assert led.format_failures() == "no recorded source registry failures"
    # No widening: the ledger is a direct-import surface, never a package export.
    assert "source_registry_failures" not in sr_mod.__all__
    assert "SourceRegistry" in sr_mod.__all__


# ── persistence / registration ─────────────────────────────────────────


def test_bind_store_failure_recorded():
    reg = _reg()
    # A raising store's row failures are swallowed by _load_persisted's
    # own debug-honest handler, so bind_store still returns True; the
    # bind_store site fires when the load step itself fails.
    with mock.patch.object(reg, "_load_persisted",
                           side_effect=RuntimeError("boom")):
        assert reg.bind_store(_RaisingStore()) is False
    assert "SourceRegistry.bind_store" in _sites()


def test_load_persisted_corrupt_stats_row_skipped_and_recorded():
    reg = _reg()
    good = _desc("good.example")
    reg.register(good)
    reg._store = SimpleNamespace(
        get=lambda key: {
            "sources": [], "candidates": [], "stats": {
                "ng|en|good.example": {"ok": 3, "fail": 1},
                "ng|en|bad.example": {"ok": "not-a-number"},
            },
        })
    reg._load_persisted()
    # Good row loaded, corrupt row skipped (not fatal, not silent).
    assert reg._stats[good.key].ok == 3
    assert "SourceRegistry._load_persisted:stats_row" in _sites()


def test_clear_persisted_failure_recorded():
    reg = _reg()
    reg._store = _RaisingStore()
    assert reg.clear_persisted() is False
    assert "SourceRegistry.clear_persisted" in _sites()


def test_register_failure_recorded():
    reg = _reg()
    assert reg.register(_BadKey()) is False
    assert "SourceRegistry.register" in _sites()
    # A healthy registration records nothing.
    clear_failures()
    assert reg.register(_desc()) is True
    assert recent_failures() == []


# ── lookup / stats / origin ────────────────────────────────────────────


def test_for_url_failure_recorded():
    reg = _reg()
    reg.register(_desc())
    with mock.patch.object(reg, "for_domain", side_effect=RuntimeError("boom")):
        assert reg.for_url("https://example.com/x") is None
    assert "SourceRegistry.for_url" in _sites()


def test_stats_for_failure_recorded():
    reg = _reg()
    stats = reg.stats_for(_BadKey())
    assert stats["ok"] == 0 and stats["success_rate"] == 0.5
    assert "SourceRegistry.stats_for" in _sites()


def test_origin_for_none_reports_registered_and_records():
    reg = _reg()
    assert reg.origin_for(None) == "registered"
    assert "SourceRegistry.origin_for" in _sites()


# ── outcome recording ──────────────────────────────────────────────────


@pytest.mark.parametrize("method,site,args", [
    ("record_no_result", "SourceRegistry.record_no_result", ()),
    ("record_blocked", "SourceRegistry.record_blocked", ()),
    ("record_rate_limited", "SourceRegistry.record_rate_limited", ()),
    ("record_outcome", "SourceRegistry.record_outcome", (True,)),
])
def test_record_methods_failure_recorded(method, site, args):
    reg = _reg()
    desc = _desc()
    reg.register(desc)
    with mock.patch.object(reg, "_persist", side_effect=RuntimeError("boom")):
        getattr(reg, method)(desc, *args)
    assert site in _sites()


# ── learning ───────────────────────────────────────────────────────────


def test_learn_rule_failure_recorded():
    reg = _reg()
    desc = _desc()
    reg.register(desc)
    with mock.patch.object(reg, "_persist", side_effect=RuntimeError("boom")):
        assert reg.learn_rule(desc, stage="direct", extraction_rule="direct") is False
    assert "SourceRegistry.learn_rule" in _sites()


def test_learned_rule_failure_recorded():
    reg = _reg()
    assert reg.learned_rule(_BadKey()) is None
    assert "SourceRegistry.learned_rule" in _sites()


def test_learned_for_url_failure_recorded():
    reg = _reg()
    with mock.patch.object(reg, "for_url", side_effect=RuntimeError("boom")):
        assert reg.learned_for_url("https://example.com/x") is None
    assert "SourceRegistry.learned_for_url" in _sites()


def test_effective_rule_failures_fall_back_and_record():
    reg = _reg()
    desc = _desc()
    reg.register(desc)
    # No hand-written rule: the learned lookup fails, the fallback is ""
    # (the stale-value shape from #117 — the learned rule is lost).
    with mock.patch.object(reg, "active_learned_rule",
                           side_effect=RuntimeError("boom")):
        assert reg.effective_extraction_rule(desc) == ""
        assert reg.effective_quality_map(desc) == ()
    assert "SourceRegistry.effective_extraction_rule" in _sites()
    assert "SourceRegistry.effective_quality_map" in _sites()


def test_note_rule_failure_recorded():
    reg = _reg()
    desc = _desc()
    reg.register(desc)
    assert reg.learn_rule(desc, stage="direct", extraction_rule="direct") is True
    with mock.patch.object(reg, "_persist", side_effect=RuntimeError("boom")):
        assert reg.note_rule_failure(desc) == ""
    assert "SourceRegistry.note_rule_failure" in _sites()


# ── scoring / ordering / intake ────────────────────────────────────────


def test_score_failure_returns_tier_default_and_records():
    reg = _reg()
    desc = _desc()
    reg.register(desc)
    with mock.patch.object(reg, "_trust", SimpleNamespace(score=_boom)):
        from nomorals.media.source_registry import _TIER_WEIGHT
        assert reg.score(desc) == _TIER_WEIGHT.get(desc.trust_tier, 0.45)
    assert "SourceRegistry.score" in _sites()


def test_ordered_for_failure_returns_unordered_pool_and_records():
    reg = _reg()
    a, b = _desc("a.example"), _desc("b.example")
    reg.register(a)
    reg.register(b)
    with mock.patch.object(reg, "health_for", side_effect=RuntimeError("boom")):
        assert reg.ordered_for() == [a, b]
    assert "SourceRegistry.ordered_for" in _sites()


def test_note_candidate_failure_recorded():
    reg = _reg()
    with mock.patch.object(reg, "_persist", side_effect=RuntimeError("boom")):
        assert reg.note_candidate("fresh.example") is False
    assert "SourceRegistry.note_candidate" in _sites()


# ── status ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method,site", [
    ("music_status", "SourceRegistry.music_status"),
    ("book_status", "SourceRegistry.book_status"),
    ("web_search_status", "SourceRegistry.web_search_status"),
    ("image_status", "SourceRegistry.image_status"),
    ("film_status", "SourceRegistry.film_status"),
    ("wisdom_status", "SourceRegistry.wisdom_status"),
])
def test_domain_status_failure_recorded(method, site):
    reg = _reg()
    with mock.patch.object(reg, "for_domain", side_effect=RuntimeError("boom")):
        getattr(reg, method)()
    assert site in _sites()


def test_betting_status_failure_recorded():
    reg = _reg()
    with mock.patch.object(reg, "by_kind", side_effect=RuntimeError("boom")):
        reg.betting_status()
    assert "SourceRegistry.betting_status" in _sites()


def test_status_failure_recorded():
    reg = _reg()
    with mock.patch.object(reg, "music_status", side_effect=RuntimeError("boom")):
        result = reg.status()
    assert result["total"] == 0
    assert "SourceRegistry.status" in _sites()


# ── deliberate exclusions (probe class, pinned as behaviour) ───────────


def test_host_of_malformed_input_is_empty_without_record():
    reg = _reg()
    # Invalid IPv6: urlparse().hostname raises — no host is determinable,
    # so "" IS the answer (#113 _is_still class), not a failure to record.
    assert reg._host_of("http://[::1") == ""
    assert recent_failures() == []


def test_verify_one_bad_template_probes_root_without_record():
    reg = _reg()
    desc = _desc(search_url_template="https://example.com/search?q={bogus}",
                 kinds=("web-search",))
    seen = []
    def _fetch(url):
        seen.append(url)
        return (200, b"ok")
    reg._verify_one(desc.key, desc, _fetch, 0.0)
    # The bad template falls through to the designed root probe route;
    # the probe still happens and its outcome is still judged.
    assert seen == ["https://example.com/"]
    assert recent_failures() == []


# ── scan pin: class remainder is exactly the honest + deliberate sites ─


def test_scan_class_remainder_is_honest_and_deliberate_only():
    from nomorals.tools import error_scan as es
    path = sr_mod.__file__
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([path])
    finally:
        import importlib
        importlib.reload(es)
    tree = ast.parse(open(path).read())
    # Enclosing SourceRegistry method per handler line.
    method_of = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SourceRegistry":
            for m in node.body:
                if isinstance(m, ast.FunctionDef):
                    for t in ast.walk(m):
                        if isinstance(t, ast.Try):
                            for h in t.handlers:
                                method_of[h.lineno] = m.name
    class_findings = [(f.rule, method_of.get(f.line))
                      for f in report.findings if f.line in method_of]
    # 3 debug-honest (exc_info=True traceback in the body) + 2 deliberate.
    assert sorted(class_findings) == sorted([
        ("E104", "_persist"),
        ("E104", "_load_persisted"),
        ("E104", "verify_health"),
        ("E104", "_host_of"),
        ("E104", "_verify_one"),
    ])
    for name in ("_persist", "_load_persisted", "verify_health"):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                src = ast.get_source_segment(open(path).read(), node)
                assert "exc_info=True" in src
