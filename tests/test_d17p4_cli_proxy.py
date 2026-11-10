"""Doc 17 Pass 4 — the ``nm proxy`` CLI surface (gap G2).

Pins:
(a) registration: the parser grows a ``proxy`` command with the
    ``sources sync`` + ``receipts`` shapes, dispatch resolves it,
    and the generated help/overview list it (the house command
    center machinery — no hand-maintained listing anywhere).
(b) ``nm proxy sources sync`` runs the SAME activation path the
    refresh seam runs (ProxyLab.activate_sources): a fresh home
    merges exactly the 15, prints the run's truth, exit 0; a
    second run is idempotent (0 added, status current).
(c) sync over a corrupt registry reports the recovery (the
    preserved backup is named) and still lands the merge.
(d) ``nm proxy receipts`` renders the P2 journal: consumers,
    routes, reasons, folded outcomes — and neither credentials
    nor internal decision/proxy ids ever print, in text or JSON.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from nomorals.cli import _cli_command_help, _cli_overview, _parser
from nomorals.cmdline import _COMMANDS
from nomorals.cmdline.commands.proxy import _cmd_proxy
from nomorals.core.proxy_routing import RouteAuthority, RouteReceiptJournal
from nomorals.tools import proxy_sources
from nomorals.tools.proxysources import SourceRegistry

EXPECTED_15 = {
    "proxifly-post", "proxifly-github-all", "proxifly-github-http",
    "proxifly-github-socks4", "proxifly-github-socks5",
    "proxmint-api", "proxmint-github-all", "proxmint-github-socks5",
    "proxmint-github-elite", "proxmint-github-json",
    "sunny9577-socks5", "proxynova", "proxy-list.download",
    "geonode-site", "proxyscrape-site",
}


def _ctx(home: Path) -> SimpleNamespace:
    return SimpleNamespace(settings=SimpleNamespace(home=str(home)),
                           db=None)


def _run(ctx: SimpleNamespace, **fields) -> tuple[int, str]:
    base = {"proxy_action": "", "proxy_sources_action": "",
            "limit": 10, "consumer": "", "source": "", "json": False}
    base.update(fields)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_proxy(argparse.Namespace(**base), ctx)
    return rc, buf.getvalue()


def _sync(ctx: SimpleNamespace) -> tuple[int, str]:
    return _run(ctx, proxy_action="sources",
                proxy_sources_action="sync")


# ── (a) registration ─────────────────────────────────────────────────────────

def test_parser_and_dispatch_register_proxy():
    parser = _parser()
    args = parser.parse_args(["proxy", "sources", "sync"])
    assert args.command == "proxy"
    assert args.proxy_action == "sources"
    assert args.proxy_sources_action == "sync"
    args = parser.parse_args(["proxy", "receipts", "--limit", "3",
                              "--consumer", "core_http"])
    assert args.proxy_action == "receipts"
    assert args.limit == 3
    assert args.consumer == "core_http"
    assert "proxy" in _COMMANDS


def test_help_surfaces_list_proxy():
    page = _cli_command_help("proxy")
    assert page is not None
    assert "sources sync" in page
    assert "proxy" in _cli_overview()


# ── (b) sources sync ─────────────────────────────────────────────────────────

def test_sync_activates_a_fresh_home_and_reports_the_run(tmp_path):
    home = tmp_path / "home"
    rc, out = _sync(_ctx(home))

    assert rc == 0
    assert "status: activated · epoch 1" in out
    assert "added: 15 source(s)" in out
    assert "already present: 14" in out
    assert "excluded: 3 — unverified or no URL" in out
    assert "registry: 101 → 116 sources" in out
    assert "+ proxifly-post (POST)" in out
    registry = SourceRegistry(home / "proxies" / "sources.json")
    assert len(registry.sources()) == 116
    record = proxy_sources.read_activation_record(registry.path)
    assert record is not None and record["epoch"] == 1
    assert set(record["last_run"]["added"]) == EXPECTED_15


def test_sync_is_idempotent_and_says_zero_added(tmp_path):
    ctx = _ctx(tmp_path / "home")
    rc, _ = _sync(ctx)
    assert rc == 0

    rc, out = _sync(ctx)

    assert rc == 0
    assert "status: current" in out
    assert "added: 0 source(s)" in out
    assert "registry: 116 → 116 sources" in out


def test_sync_json_payload_is_structured(tmp_path):
    ctx = _ctx(tmp_path / "home")
    rc, out = _run(ctx, proxy_action="sources",
                   proxy_sources_action="sync", json=True)
    assert rc == 0
    payload = json.loads(out)
    assert payload["activation"]["status"] == "activated"
    assert set(payload["activation"]["added"]) == EXPECTED_15
    assert payload["activation_record"]["epoch"] == 1
    assert payload["activation_record"]["catalog"] == "SOURCES_2026"


# ── (c) corrupt-registry sync ────────────────────────────────────────────────

def test_sync_over_a_corrupt_registry_reports_the_recovery(tmp_path):
    home = tmp_path / "home"
    proxies = home / "proxies"
    proxies.mkdir(parents=True)
    garbage = b'{"broken": [1, 2,,, not json'
    (proxies / "sources.json").write_bytes(garbage)

    rc, out = _sync(_ctx(home))

    assert rc == 0
    assert "corrupt registry recovered" in out
    assert "sources.json.corrupt-" in out
    backups = sorted(proxies.glob("sources.json.corrupt-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == garbage
    assert backups[0].name in out
    registry = SourceRegistry(proxies / "sources.json")
    assert len(registry.sources()) == 116


# ── (d) receipts ─────────────────────────────────────────────────────────────

def _seed_journal(home: Path) -> tuple[RouteAuthority, list]:
    journal = RouteReceiptJournal(home / "proxies"
                                  / "route_receipts.jsonl")
    authority = RouteAuthority(journal)
    used = authority.record(
        consumer="core_http", route="free",
        reason_code="strategy:round_robin", host="example.com",
        source="proxylab", proxy_id="10.0.0.1:8080")
    authority.record(
        consumer="downloader", route="paid",
        reason_code="healthy_order", host="cdn.example",
        source="proxypool", proxy_id="10.0.0.2:8080")
    authority.record(
        consumer="browser", route="none",
        reason_code="pool_no_route", reason="no eligible proxies",
        host="bank.example", source="proxypool")
    authority.record_outcome(used.decision_id, "used")
    return authority, [used]


def test_receipts_render_decisions_outcomes_and_no_ids(tmp_path):
    home = tmp_path / "home"
    _, receipts = _seed_journal(home)

    rc, out = _run(_ctx(home), proxy_action="receipts")

    assert rc == 0
    assert "3 decision(s)" in out
    assert "core_http" in out and "example.com" in out
    assert "strategy:round_robin" in out
    assert "outcome: used" in out
    assert "NO ROUTE" in out and "pool_no_route" in out
    for receipt in receipts:
        assert receipt.decision_id not in out
    assert "10.0.0.1:8080" not in out  # proxy ids never print


def test_receipts_consumer_filter(tmp_path):
    home = tmp_path / "home"
    _seed_journal(home)

    rc, out = _run(_ctx(home), proxy_action="receipts",
                   consumer="core_http")

    assert rc == 0
    assert "1 decision(s)" in out
    assert "example.com" in out
    assert "cdn.example" not in out


def test_receipts_empty_journal_is_an_honest_note(tmp_path):
    rc, out = _run(_ctx(tmp_path / "home"), proxy_action="receipts")
    assert rc == 0
    assert "no route decisions recorded" in out


def test_receipts_json_drops_internal_ids(tmp_path):
    home = tmp_path / "home"
    _seed_journal(home)

    rc, out = _run(_ctx(home), proxy_action="receipts", json=True)

    assert rc == 0
    payload = json.loads(out)
    assert payload["count"] == 3
    for row in payload["receipts"]:
        assert "decision_id" not in row
        assert "proxy_id" not in row
    assert payload["receipts"][0]["reason_code"] == \
        "strategy:round_robin"
    assert payload["receipts"][0]["outcome"] == "used"
