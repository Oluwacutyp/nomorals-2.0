"""Pins for Pass #118: the proxylab failure ledger.

Every converted silent handler in ``nomorals/tools/proxylab.py`` must
leave a trace in ``proxylab_failures`` while keeping its return value,
control flow, and never-raises contract. The three deliberate
exclusions are pinned as behavior (probe fall-through / expected-absent),
not comments.
"""

import ast
import socket
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.tools import proxylab
from nomorals.tools import proxylab_failures
from nomorals.tools.proxylab import (
    DomainAffinityManager,
    Proxy,
    ProxyLab,
    ProxyRotationManager,
    ProxyScraper,
    ProxyStore,
    ProxyTester,
    empty_pool_guidance,
    retry_after_delay,
)
from nomorals.tools.proxylab_failures import (
    Failure,
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def sites():
    return [f.site for f in recent_failures(100)]


def _raise(exc):
    def _f(*args, **kwargs):
        raise exc

    return _f


# ---------------------------------------------------------------- ledger


def test_ledger_record_and_recent_newest_first():
    record_failure("site.a", ValueError("first"))
    record_failure("site.b", RuntimeError("second"))
    entries = recent_failures(10)
    assert [e.site for e in entries] == ["site.b", "site.a"]
    assert entries[0].exc_type == "RuntimeError"
    assert entries[0].message == "second"
    assert isinstance(entries[0], Failure)
    assert failure_count() == 2


def test_ledger_clear_resets_count_and_ring():
    record_failure("site.a", ValueError("x"))
    assert failure_count() == 1
    clear_failures()
    assert failure_count() == 0
    assert recent_failures() == []


def test_ledger_bound_keeps_newest_200():
    for i in range(250):
        record_failure(f"site.{i}", ValueError(str(i)))
    assert failure_count() == 250
    entries = recent_failures(500)
    assert len(entries) == 200
    assert entries[0].site == "site.249"
    assert entries[-1].site == "site.50"


def test_ledger_unprintable_exc_is_safe():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("site.bad", BadStr())
    entry = recent_failures(1)[0]
    assert entry.exc_type == "BadStr"
    assert entry.message.startswith("<unprintable ")


def test_ledger_format_render_and_empty():
    assert format_failures() == "no recorded proxylab failures"
    record_failure("site.a", ValueError("boom"))
    rendered = format_failures()
    assert "site.a" in rendered
    assert "ValueError" in rendered
    assert "boom" in rendered


def test_direct_import_surface():
    assert callable(record_failure)
    assert callable(recent_failures)
    assert callable(failure_count)
    assert callable(clear_failures)
    assert callable(format_failures)
    assert proxylab_failures.__all__ == [
        "Failure", "record_failure", "recent_failures",
        "failure_count", "clear_failures", "format_failures",
    ]


def test_proxylab_all_unchanged():
    assert proxylab.__all__ == [
        "Proxy", "ProxyScraper", "ProxyTester", "ProxyStore", "ProxyLab",
        "ProxyRotationManager", "DomainAffinityManager", "default_sources",
        "rank_proxies", "score_proxy", "empty_pool_guidance", "register",
        "proxy_discover", "proxy_sources", "set_scheduler_factory",
    ]


# ---------------------------------------------------------------- sites


def test_detect_profile_settings_recorded(monkeypatch):
    monkeypatch.setenv("PREFIX", "")

    class BoomSettings:
        @property
        def profile(self):
            raise RuntimeError("settings boom")

    assert proxylab.detect_profile(SimpleNamespace(settings=BoomSettings())) \
        == "workstation"
    assert sites() == ["detect_profile:settings"]


def test_detect_profile_termux_env_recorded(monkeypatch):
    import os

    class PrefixBoomEnv(dict):
        """os.environ stand-in: only the PREFIX lookup raises.

        A blanket raising mapping breaks pytest's own failure rendering
        (it reads os.environ); delegating every other key keeps the
        harness working while the PREFIX read still explodes.
        """

        def get(self, key, default=None):
            if key == "PREFIX":
                raise RuntimeError("env boom")
            return super().get(key, default)

    monkeypatch.setattr(os, "environ", PrefixBoomEnv())
    assert proxylab.detect_profile(None) == "workstation"
    assert sites() == ["detect_profile:termux_env"]


def test_connect_sockopt_recorded(monkeypatch):
    class FakeSock:
        def setsockopt(self, *args):
            raise OSError("tuning boom")

    fake = FakeSock()
    monkeypatch.setattr(socket, "create_connection",
                        lambda *args, **kwargs: fake)
    assert proxylab._connect("example.com", 80, 5.0) is fake
    assert sites() == ["_connect:sockopt"]


def _stub_proxysources(monkeypatch, search_result):
    stub = types.ModuleType("nomorals.tools.proxysources")
    stub.DISCOVERY_SEEDS = []
    stub.github_search_repos = lambda days, limit, fetch: search_result
    stub.extract_list_candidates = (
        lambda seed, text, repo_limit, url_limit: {"github": [], "urls": []})
    monkeypatch.setitem(sys.modules, "nomorals.tools.proxysources", stub)


def test_discover_repo_listing_recorded(monkeypatch):
    _stub_proxysources(monkeypatch, ["someuser/somerepo"])
    scraper = ProxyScraper(fetcher=_raise(RuntimeError("fetch boom")),
                           profile="workstation")
    result = scraper.discover()
    assert result["registered"] == []
    assert sites() == ["ProxyScraper.discover:repo_listing"]


def test_discover_endpoint_fetch_recorded(monkeypatch):
    _stub_proxysources(monkeypatch, ["someuser/somerepo"])

    def fetcher(url):
        if "api.github.com" in url:
            return b'[{"name": "proxies.txt"}]'
        raise RuntimeError("endpoint boom")

    scraper = ProxyScraper(fetcher=fetcher, profile="workstation")
    result = scraper.discover()
    assert result["registered"] == []
    assert sites() == ["ProxyScraper.discover:endpoint_fetch"]


def test_local_ip_recorded(monkeypatch):
    tester = ProxyTester(profile="workstation")
    monkeypatch.setattr(tester, "_direct_get",
                        _raise(RuntimeError("direct boom")))
    assert tester._local_ip() == ""
    assert sites() == ["ProxyTester._local_ip"]


def test_one_local_ip_recorded(monkeypatch):
    tester = ProxyTester(profile="workstation")
    monkeypatch.setattr(tester, "_probe",
                        lambda proxy, url, marker: (200, {}, "1.2.3.4"))
    monkeypatch.setattr(tester, "_local_ip",
                        _raise(RuntimeError("local boom")))
    monkeypatch.setattr(tester, "_maybe_country", lambda proxy: None)
    proxy = Proxy(host="1.2.3.4", port=8080, scheme="http")
    result = tester.test_one(proxy)
    assert result.alive is True
    assert sites() == ["ProxyTester.test_one:local_ip"]


def test_one_anonymity_recorded(monkeypatch):
    tester = ProxyTester(profile="workstation")
    calls = {"n": 0}

    def probe(proxy, url, marker):
        calls["n"] += 1
        if calls["n"] == 1:
            return (200, {}, "1.2.3.4")
        raise RuntimeError("echo boom")

    monkeypatch.setattr(tester, "_probe", probe)
    monkeypatch.setattr(tester, "_local_ip", lambda: "5.6.7.8")
    monkeypatch.setattr(tester, "_maybe_country", lambda proxy: None)
    proxy = Proxy(host="1.2.3.4", port=8080, scheme="http")
    result = tester.test_one(proxy)
    assert result.alive is True
    assert result.anonymity == "unverified"
    assert sites() == ["ProxyTester.test_one:anonymity"]


def test_maybe_country_recorded(monkeypatch):
    tester = ProxyTester(profile="workstation", detect_country=True,
                         country_budget=60)
    monkeypatch.setattr(tester, "_probe",
                        _raise(RuntimeError("country boom")))
    proxy = Proxy(host="1.2.3.4", port=8080, scheme="http")
    tester._maybe_country(proxy)
    assert proxy.country == ""
    assert sites() == ["ProxyTester._maybe_country"]


def test_affinity_load_recorded(tmp_path):
    path = tmp_path / "affinity.json"
    path.write_text("not json{", encoding="utf-8")
    mgr = DomainAffinityManager(path)
    assert mgr._data == {}
    assert sites() == ["DomainAffinityManager._load"]


def test_affinity_save_recorded(tmp_path):
    mgr = DomainAffinityManager(tmp_path / "affinity.json")
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    mgr.path = blocker / "affinity.json"
    mgr._save()
    assert sites() == ["DomainAffinityManager._save"]


def test_empty_pool_guidance_recorded():
    class BoomStore:
        def load_working(self):
            raise RuntimeError("store boom")

    guidance = empty_pool_guidance(BoomStore())
    assert isinstance(guidance, str) and guidance
    assert sites() == ["empty_pool_guidance"]


def _lab_context(tmp_path):
    return SimpleNamespace(settings=SimpleNamespace(home=str(tmp_path)))


def test_health_last_refresh_recorded(tmp_path):
    lab = ProxyLab(_lab_context(tmp_path))
    report = lab.health()
    assert report["pool"]["last_refresh"] == "never"
    assert "ProxyLab.health:last_refresh" in sites()


def test_health_rotation_affinity_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(proxylab, "ProxyRotationManager",
                        _raise(RuntimeError("rotation boom")))
    proxylab.set_scheduler_factory(
        lambda ctx: SimpleNamespace(list_jobs=lambda **kwargs: []))
    try:
        lab = ProxyLab(_lab_context(tmp_path))
        lab.health()
    finally:
        proxylab.set_scheduler_factory(None)
    # last_refresh also records on this bare store (working_json absent);
    # the isolation claim is rotation_affinity yes, schedule no.
    assert "ProxyLab.health:rotation_affinity" in sites()
    assert "ProxyLab.health:schedule" not in sites()


def test_health_schedule_recorded(monkeypatch, tmp_path):
    class QuietRotation:
        def __init__(self, context):
            pass

        def status(self):
            return {}

    monkeypatch.setattr(proxylab, "ProxyRotationManager", QuietRotation)
    proxylab.set_scheduler_factory(
        lambda ctx: SimpleNamespace(
            list_jobs=_raise(RuntimeError("schedule boom"))))
    try:
        lab = ProxyLab(_lab_context(tmp_path))
        lab.health()
    finally:
        proxylab.set_scheduler_factory(None)
    # last_refresh also records on this bare store (working_json absent);
    # the isolation claim is schedule yes, rotation_affinity no.
    assert "ProxyLab.health:schedule" in sites()
    assert "ProxyLab.health:rotation_affinity" not in sites()


def test_retry_after_default_recorded():
    assert retry_after_delay({"Retry-After": "not-a-date-or-number"}) == 60.0
    assert sites() == ["retry_after_delay:default"]


def test_subnet_of_recorded():
    assert proxylab._subnet_of("not-an-ip") == "not-an-ip"
    assert sites() == ["_subnet_of"]


def test_rotation_load_state_recorded(monkeypatch):
    class BoomKV:
        def __init__(self, db):
            pass

        def get(self, key):
            raise RuntimeError("kv boom")

    monkeypatch.setattr(proxylab, "KVStore", BoomKV)
    mgr = ProxyRotationManager(SimpleNamespace(db=object()))
    assert mgr.state["enabled"] is False
    assert sites() == ["ProxyRotationManager._load_state"]


def test_rotation_pool_urls_recorded(monkeypatch, tmp_path):
    stub = types.ModuleType("nomorals.tools.ssh_socks")

    class BoomSocks:
        @staticmethod
        def _live_urls():
            raise RuntimeError("tunnel boom")

    stub.SshSocksManager = BoomSocks
    monkeypatch.setitem(sys.modules, "nomorals.tools.ssh_socks", stub)
    mgr = ProxyRotationManager(_lab_context(tmp_path))
    assert mgr._pool_urls() == []
    assert sites() == ["ProxyRotationManager._pool_urls"]


# ------------------------------------------------- deliberate exclusions


def test_auto_parse_probe_fallthrough_not_recorded():
    proxies, kind = ProxyScraper._auto_parse("1.2.3.4:8080\n5.6.7.8:3128\n")
    assert len(proxies) == 2
    assert kind == "list"
    assert sites() == []


def test_clear_missing_files_not_recorded(tmp_path):
    store = ProxyStore(tmp_path / "proxies")
    store.clear()
    assert sites() == []


def test_retry_after_float_to_date_probe_not_recorded():
    assert retry_after_delay(
        {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 1.0
    assert sites() == []


def test_scan_remainder_is_exactly_the_three_deliberate_sites(monkeypatch):
    from nomorals.tools import error_scan as es

    monkeypatch.setattr(es, "_noqa_suppresses",
                        lambda line, rule: False)
    path = Path(proxylab.__file__)
    report = es.scan([str(path)])
    findings = [f for f in report.findings if f.file == str(path)]
    assert len(findings) == 3
    assert {f.rule for f in findings} == {"E103"}
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def enclosing(lineno):
        best = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= lineno <= (node.end_lineno or lineno):
                    if best is None or node.lineno >= best.lineno:
                        best = node
        return best.name if best else None

    assert {enclosing(f.line) for f in findings} == {
        "_auto_parse", "clear", "retry_after_delay",
    }
