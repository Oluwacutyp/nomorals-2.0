"""Doc 17 — acceptance battery: the programme's whole story, in
one offline narrative over one temp home.

    fresh registry (101) → CLI sync activates (116, epoch 1,
    exactly the 15 + 3 excluded-with-reasons) → a core.http-shaped
    fetch decision mints a receipt with the machinery's reason →
    a downloader-shaped decision mints its receipts and its
    outcomes fold → a corrupt sources.json is quarantined
    byte-for-byte, the registry recovers, the record carries the
    recovery → the /proxy render shows activation + receipts
    truthfully → a CLI sync on the current registry reports 0
    added → a no-route decision mints its none-receipt and renders
    as one → the WS4 board line (read-only here) tells the same
    truth throughout.

Everything runs on fake fetchers/pools against the temp home —
no network, no public proxies.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import nomorals.tools.proxylab as PL
from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.cmdline.commands.proxy import _cmd_proxy
from nomorals.connectors.proxypool import ProxyPoolError
from nomorals.core.proxy_routing import (
    RouteAuthority,
    RouteReceiptJournal,
)
from nomorals.media.downloader import MediaDownloader
from nomorals.media.explore.board import PROXY_DEPENDENCY_LINE, proxy_line
from nomorals.tools import proxy_sources
from nomorals.tools.proxylab_failures import clear_failures
from nomorals.tools.proxysources import SourceRegistry

EXPECTED_15 = {
    "proxifly-post", "proxifly-github-all", "proxifly-github-http",
    "proxifly-github-socks4", "proxifly-github-socks5",
    "proxmint-api", "proxmint-github-all", "proxmint-github-socks5",
    "proxmint-github-elite", "proxmint-github-json",
    "sunny9577-socks5", "proxynova", "proxy-list.download",
    "geonode-site", "proxyscrape-site",
}
DL_PASSWORD = "<redacted>"


def _cli(ctx, **fields) -> tuple[int, str]:
    base = {"proxy_action": "", "proxy_sources_action": "",
            "limit": 10, "consumer": "", "source": "", "json": False}
    base.update(fields)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_proxy(argparse.Namespace(**base), ctx)
    return rc, buf.getvalue()


def _sync(ctx) -> tuple[int, str]:
    return _cli(ctx, proxy_action="sources",
                proxy_sources_action="sync")


class _FakeTools:
    def call(self, name: str, **kwargs):
        assert name == "proxy_status"
        return SimpleNamespace(
            ok=True, value={"known": [], "active": "direct",
                            "raw_sockets_routed": False})


def test_doc17_programme_end_to_end(tmp_path):
    home = tmp_path / "home"
    proxies_dir = home / "proxies"
    proxies_dir.mkdir(parents=True)
    ctx = SimpleNamespace(settings=SimpleNamespace(home=str(home)),
                          db=None)
    journal = RouteReceiptJournal(proxies_dir
                                  / "route_receipts.jsonl")
    clear_failures()

    # — 1 · a fresh registry: 101 built-ins, no activation record,
    #        the board still tells the not-yet truth.
    registry = SourceRegistry(proxies_dir / "sources.json")
    assert len(registry.sources()) == 101
    assert proxy_sources.activation_state_for_home(home) is None
    assert proxy_line(ctx) == PROXY_DEPENDENCY_LINE

    # — 2 · CLI sync activates: exactly the 15 land, 3 excluded with
    #        reasons, epoch 1 is recorded, the board flips.
    rc, out = _sync(ctx)
    assert rc == 0
    assert "status: activated · epoch 1" in out
    assert "added: 15 source(s)" in out
    assert "excluded: 3 — unverified or no URL" in out
    assert "registry: 101 → 116 sources" in out
    record = proxy_sources.activation_state_for_home(home)
    assert record is not None and record["epoch"] == 1
    assert set(record["last_run"]["added"]) == EXPECTED_15
    assert {e["name"] for e in
            record["last_run"]["excluded_unverified"]} == {
                "hidemy-name", "openproxy.space", "scrappey"}
    registry = SourceRegistry(proxies_dir / "sources.json")
    assert len(registry.sources()) == 116
    board = proxy_line(ctx)
    assert "epoch 1" in board and "registry holds 116 sources" in board

    # — 3 · a core.http-shaped fetch decision (the rotation
    #        resolver is what core.http calls per request) mints a
    #        receipt carrying the machinery's own reason; a
    #        reported failure folds back as its outcome.
    lab = PL.ProxyLab(ctx)
    now = time.time()
    pool_proxies = [
        PL.Proxy(host="10.0.0.1", port=8080, scheme="http",
                 anonymity="elite", country="NG", alive=True,
                 tested_at=now),
        PL.Proxy(host="10.0.0.2", port=1080, scheme="socks5",
                 anonymity="anonymous", country="DE", alive=True,
                 tested_at=now),
    ]
    for p in pool_proxies:
        p.record_latency(120.0, at=now)
    lab.store.save(pool_proxies)
    rot = PL.ProxyRotationManager(ctx)
    rot.state["enabled"] = True
    url = rot._resolver("https://example.com/x")  # noqa: SLF001
    assert url
    rows = journal.recent(consumer="core_http")
    assert len(rows) == 1
    assert rows[0]["route"] == "free"
    assert rows[0]["reason_code"].startswith("strategy:")
    # the strategy pick is target-less by design (the resolver
    # records the domain only when affinity binds) — the receipt
    # carries no host here, and the surfaces render that honestly.
    assert rows[0]["host"] == ""
    assert rows[0]["outcome"] is None
    rot.report_error(url, "connection reset")
    rows = journal.recent(consumer="core_http")
    assert rows[0]["outcome"] == "failed"
    assert rows[0]["outcome_detail"] == "connection reset"

    # — 4 · a downloader-shaped decision chain: two pool attempts,
    #        two receipts; the failed fetch folds "failed", the
    #        landed file folds "used" — and the pool password
    #        never reaches the journal bytes.
    dl_ctx = MagicMock()
    dl_ctx.home = home
    dl_ctx.db = MagicMock()
    dl_ctx.settings = None
    dl = MediaDownloader(dl_ctx, direct_retries=1, max_proxies=3)
    audio = home / "p.mp3"
    audio.write_bytes(b"data")

    def pool_row(pid, port):
        return {"id": pid, "protocol": "https", "host": "10.0.0.9",
                "port": port,
                "proxy_url": f"https://user:{DL_PASSWORD}@10.0.0.9:{port}",
                "url": f"https://10.0.0.9:{port}"}

    pool = MagicMock()
    pool.healthy_proxies.return_value = [pool_row("dl-1", 8081),
                                         pool_row("dl-2", 8082)]

    def fake_download(url_, dest, **kwargs):
        proxy = kwargs.get("proxy", "")
        if not proxy:
            raise RuntimeError("connection reset by peer")
        if "8081" in proxy:
            raise RuntimeError("connection refused")
        return {"url": url_, "path": str(audio), "bytes": 4,
                "title": "t", "extractor": "yt_dlp", "seconds": 0.1}

    env_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy")
    with patch.dict(os.environ, {k: "" for k in env_keys}), \
            patch("nomorals.tools.media.download",
                  side_effect=fake_download), \
            patch.object(dl, "_proxy_pool", return_value=pool), \
            patch.object(dl, "_stage_browser", return_value=False):
        report = dl.download("https://x.test/a.mp3")
    assert report.ok
    dl_rows = journal.recent(consumer="downloader")
    assert [r["reason_code"] for r in dl_rows] == ["healthy_order",
                                                   "healthy_order"]
    assert dl_rows[0]["outcome"] == "failed"
    assert dl_rows[1]["outcome"] == "used"
    assert DL_PASSWORD.encode() not in \
        (proxies_dir / "route_receipts.jsonl").read_bytes()

    # — 5 · corrupt the registry in place: the next sync preserves
    #        the garbage byte-for-byte, rebuilds, re-merges, and
    #        the record carries the recovery.
    garbage = b'{"broken": [1, 2,,, not json'
    (proxies_dir / "sources.json").write_bytes(garbage)
    rc, out = _sync(ctx)
    assert rc == 0
    assert "corrupt registry recovered" in out
    backups = sorted(proxies_dir.glob("sources.json.corrupt-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == garbage
    assert backups[0].name in out
    record = proxy_sources.activation_state_for_home(home)
    assert record["last_run"]["corrupt_recovery"]["backup"] == \
        backups[0].name
    registry = SourceRegistry(proxies_dir / "sources.json")
    assert len(registry.sources()) == 116

    # — 6 · the /proxy status render shows both blocks truthfully:
    #        activation (with the recovery note) + receipts from
    #        both consumers — and no internal ids anywhere.
    rt = RuntimeSystemMixin()
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(home=str(home)), tools=_FakeTools())
    clear_failures()
    text = rt._control_proxy("status")
    assert f"catalogue activation: epoch {record['epoch']}" in text
    assert "registry holds 116 sources" in text
    assert "a corrupt registry was recovered on the way in" in text
    assert "core_http → free" in text
    assert "strategy:round_robin" in text
    assert "downloader" in text and "healthy_order" in text
    assert "outcome: failed" in text and "outcome: used" in text
    raw_rows = [line for line in
                (proxies_dir / "route_receipts.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in raw_rows:
        row = json.loads(line)
        if row.get("decision_id"):
            assert row["decision_id"] not in text
    assert DL_PASSWORD not in text

    # — 7 · a CLI sync on the now-current registry: 0 added.
    rc, out = _sync(ctx)
    assert rc == 0
    assert "status: current" in out
    assert "added: 0 source(s)" in out

    # — 8 · a no-route decision mints its none-receipt and renders
    #        as one, on the CLI and in chat.
    class _EmptyPool:
        def select(self, **filters):
            raise ProxyPoolError("no eligible proxies")

        def healthy_proxies(self):
            return []

    authority = RouteAuthority(journal)
    try:
        authority.decide_pool(_EmptyPool(), consumer="browser",
                              host="bank.example")
        raise AssertionError("empty pool must raise")
    except ProxyPoolError:
        pass
    none_rows = [r for r in journal.recent(consumer="browser")
                 if r["route"] == "none"]
    assert len(none_rows) == 1
    assert none_rows[0]["reason_code"] == "pool_no_route"
    rc, out = _cli(ctx, proxy_action="receipts", limit=50)
    assert rc == 0
    assert "NO ROUTE" in out and "bank.example" in out
    assert "pool_no_route" in out
    assert none_rows[0]["decision_id"] not in out

    # — 9 · the board line still tells the activated truth.
    assert "the source catalogue is activated (doc 17)" in \
        proxy_line(ctx)
