"""Doc 17 Pass 1 — transactional proxy-source activation.

Pins, end to end:

(a) A fresh registry + one activation merges EXACTLY the 15
    verified-with-URL catalogue entries the evaluation reproduced
    offline (101 built-ins → 116), writes the epoch record beside
    the registry, and is idempotent on the second cycle.
(b) The catalogue's one POST feed (proxifly-post) carries its
    method catalogue → registry record → scraper: the scraper
    issues the fetch AS a POST through the fetch_proxifly
    adapter, never through its GET fetcher, so the source's
    health record cannot be poisoned by a wrong-method fetch.
(c) A non-GET source with no adapter is never fetched and never
    health-recorded: it surfaces as ``unsupported_kind``.
(d) A corrupt sources.json is preserved byte-for-byte in a
    named ``.corrupt-*.bak`` sibling (never silently
    overwritten), the event lands in the failure ledger + the
    activation record, and the registry rebuilds functional; a
    valid file is never backed up.
(e) Activation is fenced: a held lock makes a second activation
    a named skip that changes nothing; a stale lock is stolen.
(f) A merge that leaves the registry empty is a NAMED empty
    result, recorded — never a silent no-op.
(g) The /explore board's proxy line renders the activation
    record: the not-yet-activated form before the first run, the
    recorded epoch/merge/registry size after it.
(h) ProxyLab.refresh — the seam the durable ``proxy_refresh``
    job fires — activates first and reports it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.media.explore.board import (
    PROXY_DEPENDENCY_LINE,
    proxy_line,
    render_explore_board,
)
from nomorals.media.explore.store import ExploreStore
from nomorals.tools import proxy_sources, proxysources
from nomorals.tools.proxylab import ProxyLab, ProxyScraper
from nomorals.tools.proxylab_failures import clear_failures, recent_failures
from nomorals.tools.proxysources import SourceRegistry

#: The evaluation's offline reproduction (doc17-eval §1): the 15
#: verified-with-URL catalogue entries absent from a fresh
#: registry. Activation must merge exactly these — no more, no
#: fewer.
EXPECTED_15 = {
    "proxifly-post", "proxifly-github-all", "proxifly-github-http",
    "proxifly-github-socks4", "proxifly-github-socks5",
    "proxmint-api", "proxmint-github-all", "proxmint-github-socks5",
    "proxmint-github-elite", "proxmint-github-json",
    "sunny9577-socks5", "proxynova", "proxy-list.download",
    "geonode-site", "proxyscrape-site",
}

PROXIFLY_URL = "https://api.proxifly.dev/get-proxy"
POST_JSON = json.dumps({"data": [{
    "ip": "9.9.9.9", "port": 8080, "protocol": "http",
    "country": "US", "speed": 50,
}]}).encode("utf-8")


def _registry(tmp_path: Path) -> SourceRegistry:
    return SourceRegistry(tmp_path / "sources.json")


def _ctx(home: Path) -> SimpleNamespace:
    return SimpleNamespace(settings=SimpleNamespace(home=str(home)))


# ── (a) the merge + the epoch record ─────────────────────────────────────────

def test_activation_merges_exactly_the_15_and_records_the_epoch(tmp_path):
    registry = _registry(tmp_path)
    assert len(registry.sources()) == 101

    run = proxy_sources.activate_sources(registry)

    assert run["status"] == "activated"
    assert set(run["added"]) == EXPECTED_15
    assert run["registry_before"] == 101
    assert run["registry_after"] == 116
    assert run["epoch"] == 1
    # every catalogue entry is accounted for: 15 merged + 14
    # already present (verified entries overlapping the built-ins)
    # + exactly 3 excluded as unverified/no-URL, each with its
    # reason — merged vs present vs excluded fully explained.
    assert run["already_present"] == 14
    assert {e["name"] for e in run["excluded_unverified"]} == {
        "hidemy-name", "openproxy.space", "scrappey"}
    assert all(e["reason"] == "unverified_or_no_url"
               for e in run["excluded_unverified"])

    record_path = proxy_sources.activation_record_path(registry.path)
    assert record_path.exists()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["epoch"] == 1
    assert record["catalog"] == "SOURCES_2026"
    assert record["catalog_size"] == 32
    assert record["catalog_version"] == proxy_sources.catalog_version()
    assert record["first_activated_at"] == record["last_run"]["at"]
    assert set(record["last_run"]["added"]) == EXPECTED_15
    assert len(record["runs"]) == 1


def test_second_activation_is_idempotent_and_advances_the_epoch(tmp_path):
    registry = _registry(tmp_path)
    proxy_sources.activate_sources(registry)

    run = proxy_sources.activate_sources(registry)

    assert run["status"] == "current"
    assert run["added"] == []
    # 29 of the 32 catalogue entries are now present (the 15
    # merged + the 14 built-in overlaps); 3 stay excluded.
    assert run["already_present"] == 29
    assert run["registry_before"] == 116
    assert run["registry_after"] == 116
    assert run["epoch"] == 2
    record = proxy_sources.read_activation_record(registry.path)
    assert record is not None
    assert record["epoch"] == 2
    assert len(record["runs"]) == 2
    assert record["first_activated_at"] == record["runs"][0]["at"]


# ── (b) the POST method, carried end to end ──────────────────────────────────

def test_method_is_recorded_in_the_registry_and_on_disk(tmp_path):
    registry = _registry(tmp_path)
    run = proxy_sources.activate_sources(registry)

    assert run["methods"] == {"proxifly-post": "POST"}
    assert registry.methods()["proxifly-post"] == "POST"
    assert registry.methods()["proxynova"] == "GET"
    on_disk = json.loads(registry.path.read_text(encoding="utf-8"))
    assert on_disk["proxifly-post"]["method"] == "POST"
    assert on_disk["proxynova"]["method"] == "GET"


def test_scraper_fetches_the_post_entry_as_a_post_never_a_get(tmp_path):
    get_calls: list[str] = []
    post_calls: list[tuple] = []

    def fake_get(url: str) -> bytes:
        get_calls.append(url)
        return b"1.2.3.4:8080\n"

    def fake_post(url, data, headers, timeout):
        post_calls.append((url, data, headers, timeout))
        return POST_JSON

    scraper = ProxyScraper(
        fetcher=fake_get, post_fetcher=fake_post,
        methods={"proxifly-post": "POST"},
        sources=[("proxifly-post", PROXIFLY_URL, "list"),
                 ("plain-get", "https://example.com/list.txt", "list")])
    report = scraper.scrape()

    # the POST endpoint was never GET-ed…
    assert get_calls == ["https://example.com/list.txt"]
    # …it was fetched AS a POST, once, at its catalogue probe
    # budget (2), through the proxifly adapter's form.
    assert len(post_calls) == 1
    url, data, _headers, _timeout = post_calls[0]
    assert url == PROXIFLY_URL
    assert data["protocol"] == "http"
    assert int(data["quantity"]) == 2
    assert report["by_source"]["proxifly-post"] == 1
    assert report["by_source"]["plain-get"] == 1
    assert report["errors"] == {}


def test_post_entry_health_record_cannot_be_poisoned(tmp_path):
    """Lab-level: after activation, a scrape round fetches
    proxifly-post through its adapter — its registry health
    record shows the harvest, zero fails."""
    lab = ProxyLab(_ctx(tmp_path))
    lab.activate_sources()
    assert lab.registry.methods()["proxifly-post"] == "POST"

    fetched: list[str] = []

    def fake_get(url: str) -> bytes:
        fetched.append(url)
        return b"1.2.3.4:8080\n"

    lab._scraper._fetcher = fake_get
    lab._scraper._post_fetcher = (
        lambda url, data, headers, timeout: POST_JSON)
    report = lab.scrape(sources=[
        ("proxifly-post", PROXIFLY_URL, "list"),
        ("plain-get", "https://example.com/list.txt", "list")])

    assert PROXIFLY_URL not in fetched
    assert report["by_source"]["proxifly-post"] == 1
    health = {h["name"]: h for h in lab.registry.health()}
    assert health["proxifly-post"]["fails"] == 0
    assert health["proxifly-post"]["last_found"] == 1
    # health()'s last_ok is seconds-since-success, None when the
    # source has NEVER succeeded: a poisoned (wrong-method) round
    # would leave it None with fails climbing toward disable.
    assert health["proxifly-post"]["last_ok"] is not None


# ── (c) a non-GET source with no adapter ─────────────────────────────────────

def test_unsupported_non_get_source_is_never_fetched_or_recorded(tmp_path):
    fetched: list[str] = []

    def fake_get(url: str) -> bytes:
        fetched.append(url)
        return b"1.2.3.4:8080\n"

    scraper = ProxyScraper(
        fetcher=fake_get, methods={"odd-src": "DELETE"},
        sources=[("odd-src", "https://example.com/x", "list")])
    report = scraper.scrape()

    assert fetched == []
    assert report["errors"] == {}
    assert report["by_source"] == {}
    assert "odd-src" in report["unsupported"]
    assert "unsupported_kind" in report["unsupported"]["odd-src"]


# ── (d) the corrupt-file policy ──────────────────────────────────────────────

def test_corrupt_registry_is_preserved_recorded_and_rebuilt(tmp_path):
    clear_failures()
    path = tmp_path / "sources.json"
    garbage = b'{"broken": [1, 2,,, not json'
    path.write_bytes(garbage)
    registry = SourceRegistry(path)

    sources = registry.sources()  # the call that used to overwrite
    assert len(sources) == 101  # built-ins: the registry rebuilt

    backups = sorted(tmp_path.glob("sources.json.corrupt-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == garbage  # bytes preserved
    # the registry file itself is valid JSON again (the rebuild).
    assert isinstance(json.loads(path.read_text(encoding="utf-8")),
                      dict)
    recovery = registry.corrupt_recovery()
    assert recovery is not None
    assert recovery["backup"] == backups[0].name
    assert recovery["error"]
    sites = [f.site for f in recent_failures()]
    assert any("corrupt_registry" in s for s in sites)


def test_activation_over_a_corrupt_registry_records_the_recovery(tmp_path):
    path = tmp_path / "sources.json"
    path.write_bytes(b"\x89not json at all")
    registry = SourceRegistry(path)

    run = proxy_sources.activate_sources(registry)

    assert run["status"] == "activated"
    assert set(run["added"]) == EXPECTED_15
    backup = run["corrupt_recovery"]["backup"]
    assert backup
    assert (tmp_path / backup).read_bytes() == b"\x89not json at all"
    record = proxy_sources.read_activation_record(path)
    assert record["last_run"]["corrupt_recovery"]["backup"] == backup


def test_valid_registry_is_never_backed_up(tmp_path):
    registry = _registry(tmp_path)
    registry.sources()  # writes a valid file
    registry.sources()  # loads it clean
    assert registry.corrupt_recovery() is None
    assert list(tmp_path.glob("*.corrupt-*.bak")) == []
    run = proxy_sources.activate_sources(registry)
    assert run["corrupt_recovery"] == {}


# ── (e) fencing ──────────────────────────────────────────────────────────────

def test_held_lock_makes_activation_a_named_skip(tmp_path):
    registry = _registry(tmp_path)
    lock = proxy_sources._ActivationLock(registry.path)
    assert lock.acquire()
    try:
        run = proxy_sources.activate_sources(registry)
    finally:
        lock.release()
    assert run["status"] == "skipped"
    assert run["reason"] == "activation_in_progress"
    assert run["added"] == []
    # nothing merged, nothing recorded: the skip changed nothing.
    assert len(registry.sources()) == 101
    assert proxy_sources.read_activation_record(registry.path) is None


def test_stale_lock_is_stolen_and_activation_proceeds(tmp_path):
    registry = _registry(tmp_path)
    lock_path = registry.path.with_name(registry.path.name + ".lock")
    lock_path.write_text(json.dumps(
        {"token": "dead-holder", "pid": 0,
         "at": time.time() - 10_000}), encoding="utf-8")

    run = proxy_sources.activate_sources(registry)

    assert run["status"] == "activated"
    assert set(run["added"]) == EXPECTED_15
    assert not lock_path.exists()  # released after the run


# ── (f) the named empty ──────────────────────────────────────────────────────

def test_empty_catalogue_and_registry_is_a_named_empty(tmp_path):
    registry = _registry(tmp_path)
    with mock.patch.object(proxysources, "BUILT_IN_SOURCES", []), \
            mock.patch.object(proxy_sources, "SOURCES_2026", ()):
        run = proxy_sources.activate_sources(registry)
    assert run["status"] == "empty"
    assert run["added"] == []
    assert run["registry_after"] == 0
    record = proxy_sources.read_activation_record(registry.path)
    assert record is not None
    assert record["last_run"]["status"] == "empty"


# ── (g) the board line ───────────────────────────────────────────────────────

def test_board_line_before_and_after_activation(tmp_path):
    home = tmp_path / "home"
    proxies_dir = home / "proxies"
    proxies_dir.mkdir(parents=True)
    ctx = _ctx(home)

    # never activated on this host: the not-yet form, verbatim.
    assert proxy_line(ctx) == PROXY_DEPENDENCY_LINE

    registry = SourceRegistry(proxies_dir / "sources.json")
    proxy_sources.activate_sources(registry)

    line = proxy_line(ctx)
    assert "the source catalogue is activated" in line
    assert "epoch 1" in line
    assert "added 15 source(s)" in line
    assert "registry holds 116 sources" in line

    text = render_explore_board(context=ctx, store=ExploreStore())
    assert line in text


# ── (h) the refresh seam ─────────────────────────────────────────────────────

def test_lab_refresh_activates_first_and_reports_it(tmp_path):
    """Mirrors the shipped refresh-hooks test's stubbing exactly:
    the durable proxy_refresh job fires lab.refresh(); activation
    rides its head — the merge reaches the live registry with no
    parallel scheduler."""
    lab = ProxyLab(_ctx(tmp_path))
    lab._scraper = ProxyScraper(  # noqa: SLF001 - shipped test idiom
        fetcher=lambda url: b"1.2.3.4:8080\n",
        sources=[("s-ok", "http://ok/x.txt", "list")])
    lab.registry.sources = lambda: [("s-ok", "http://ok/x.txt", "list")]

    class _StubTester:
        detect_country = True

        def test_many(self, proxies, limit=150):
            return []

    lab._tester = _StubTester()  # noqa: SLF001
    events: list[tuple] = []
    report = lab.refresh(
        on_scrape_source=lambda n, a, e: events.append((n, a, e)))

    assert set(report["activation"]["added"]) == EXPECTED_15
    assert report["activation"]["registry_after"] == 116
    assert report["scraped"] == 1
    assert events == [("s-ok", 1, "")]
    record = proxy_sources.read_activation_record(lab.registry.path)
    assert record is not None and record["epoch"] == 1
