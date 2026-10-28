"""Tail-18 pins for nomorals/search/web.py — 0 converts / 10 exclusions.

Every pin asserts the documented degraded value AND containment: the
failure never raises, and no record lands anywhere it should not. The
search package has no failures ledger and web.py imports no logging at
all, so the module's only observability surface is the source registry
itself — NOT-recording is pinned against Mock registries (no successful
record call for the failed path, no cascading record attempt).
"""
import io
import types
import urllib.error
import urllib.request
from unittest import mock

import pytest

from nomorals.search import web as W


class _RaisingFP(io.RawIOBase):
    def read(self, *a, **k):
        raise OSError("fp dead")


def _http_error_403(fp):
    return urllib.error.HTTPError(
        "https://example.invalid/x", 403, "Forbidden", {}, fp)


def _raising_urlopen(url, timeout=None):
    raise _http_error_403(_RaisingFP())


class _RaisingKV:
    def __init__(self, db):
        raise RuntimeError("kv down")


class _RaisingDB:
    accessed = []

    @property
    def db(self):
        type(self).accessed.append(True)
        raise RuntimeError("db property dead")


def test_175_unreadable_error_body_returns_code_with_empty_body():
    with mock.patch.object(urllib.request, "urlopen", _raising_urlopen):
        code, body = W._http("GET", "https://example.invalid/x")
    assert (code, body) == (403, b"")
    # No fabricated body: exactly empty, not a synthesized error page.


def test_250_kv_bind_failure_still_returns_registry_unbound():
    import nomorals.media.source_registry as reg_mod
    import nomorals.storage.kv as kv_mod

    bind_calls = []
    with mock.patch.object(kv_mod, "KVStore", _RaisingKV), \
            mock.patch.object(reg_mod.DEFAULT_REGISTRY, "bind_store",
                              lambda store: bind_calls.append(store)):
        reg = W.web_registry_for_context(types.SimpleNamespace(db=object()))
    assert reg is reg_mod.DEFAULT_REGISTRY
    assert bind_calls == []  # bind never ran: no partial persistence state.


def test_254_raising_db_property_returns_none():
    _RaisingDB.accessed.clear()
    assert W.web_registry_for_context(_RaisingDB()) is None
    assert _RaisingDB.accessed == [True]  # the raise genuinely happened.


def test_292_ensure_instance_failure_returns_none_and_records_nothing():
    reg = mock.Mock()
    reg.for_domain.side_effect = RuntimeError("reg dead")
    assert W._ensure_searx_instance(reg, "https://sx.example.com") is None
    assert reg.record_outcome.call_count == 0
    assert reg.record_no_result.call_count == 0


def test_315_backend_desc_failure_returns_none_and_records_nothing():
    reg = mock.Mock()
    reg.for_domain.side_effect = RuntimeError("reg dead")
    src = types.SimpleNamespace(name="web_brave")
    assert W.registry_desc_for_backend(reg, src) is None
    assert reg.record_outcome.call_count == 0


def test_394_ordering_failure_returns_input_order_and_records_nothing():
    reg = mock.Mock()
    reg.for_domain.side_effect = lambda domain: types.SimpleNamespace(
        kinds=("search", "web-search"))
    reg.stats_for.side_effect = RuntimeError("stats dead")
    names = ["web_brave", "web_tavily"]
    assert W.ordered_web_backend_names(reg, names) == names
    assert reg.record_outcome.call_count == 0
    assert reg.record_no_result.call_count == 0


def test_454_failed_credit_attempt_is_contained():
    reg = mock.Mock()
    reg.record_outcome.side_effect = RuntimeError("record dead")
    src = W.BraveWebSource(source_registry=reg)
    assert src._record_search_outcome([{"title": "x"}], 0.1) is None
    assert reg.record_outcome.call_count == 1  # the single failed attempt
    assert reg.record_no_result.call_count == 0
    assert reg.record_rate_limited.call_count == 0


def test_478_failed_blame_attempt_is_contained():
    reg = mock.Mock()
    reg.record_outcome.side_effect = RuntimeError("record dead")
    src = W.BraveWebSource(source_registry=reg)
    assert src._record_search_failure(RuntimeError("boom"), 0.1) is None
    assert reg.record_outcome.call_count == 1  # the single failed attempt
    assert reg.record_no_result.call_count == 0
    assert reg.record_rate_limited.call_count == 0


def test_658_failed_instance_credit_attempt_is_contained():
    reg = mock.Mock()
    reg.for_domain.return_value = types.SimpleNamespace(
        kinds=("search", "web-search"))
    reg.record_outcome.side_effect = RuntimeError("record dead")
    sx = W.SearXNGWebSource(source_registry=reg)
    assert sx._record_instance("https://sx.example.com", "served", 0.1) is None
    assert reg.record_outcome.call_count == 1  # the single failed attempt
    assert reg.record_no_result.call_count == 0
    assert reg.record_rate_limited.call_count == 0


def test_1037_probe_failure_reports_backend_unavailable():
    from nomorals.search.web import BraveWebSource

    real_probe = BraveWebSource.probe
    try:
        BraveWebSource.probe = lambda self: (_ for _ in ()).throw(
            RuntimeError("probe dead"))
        out = W.web_backends_configured()
    finally:
        BraveWebSource.probe = real_probe
    assert out["web_brave"] is False
    assert len(out) == 6  # no backend dropped from the report
    assert all(isinstance(v, bool) for v in out.values())
