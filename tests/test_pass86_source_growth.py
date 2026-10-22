"""Pass #86 — the source registry grows live.

Pass #76 converged downloads on ONE registry
(``media/source_registry.SourceRegistry``) but its growth machinery had
zero live callers: no outcome recording on the download path, no
persistence (in-memory only — a restart forgot everything learned), no
health loop, no discovery intake.  In behaviour the registry was still
CLOSED-SOURCE (the owner's open-sources law, 2026-10-21): seeds were
all it ever knew.

Survey truth this suite pins (the brief's "existing machinery" was
partly aspirational):

* ``record_outcome`` / ``score`` / ``ordered_for`` DID exist — with
  exactly one test caller and no production caller.
* Health / discovery / persistence did NOT exist in the registry at
  all (``_stats_path`` was a dead field).  The registry speaks neither
  agency ``SourceHealth`` nor the proxy lab; the smallest real versions
  are built in the registry itself, mirroring SourceHealth's
  vocabulary + circuit-breaker thresholds (stand-down 3, disable 10)
  and the ``KVStore`` persistence pattern ``core.trust.SourceTrust``
  already uses for the same job.

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


def _landed(url, dest, **kw):
    p = Path(dest) / "song.mp3"
    p.write_bytes(b"ID3" + b"y" * 32)
    return {"url": url, "path": str(p), "bytes": p.stat().st_size,
            "title": "Song", "extractor": "fake", "seconds": 0.1}


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


def _dead_fetcher(url):
    return 404, b"gone"


# ── (1) outcome recording is live on the download path ────────────────


def test_download_success_credits_the_serving_source(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    yt = reg.for_url("https://www.youtube.com/watch?v=abc")
    assert yt is not None
    before = reg.stats_for(yt)
    assert before["ok"] == 0 and before["fail"] == 0

    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg)
    with mock.patch("nomorals.tools.media.download", side_effect=_landed):
        report = dl.download("https://www.youtube.com/watch?v=abc",
                             dest_dir=tmp_path)
    assert report.ok, report.summary()
    after = reg.stats_for(yt)
    assert after["ok"] == 1 and after["fail"] == 0


def test_download_failure_degrades_the_source(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    pristine = _seeded()
    yt = reg.for_url("https://www.youtube.com/watch?v=abc")
    base_score = pristine.score(yt)

    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg,
                         direct_retries=1)
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("connection reset by peer")), \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        report = dl.download("https://www.youtube.com/watch?v=abc",
                             dest_dir=tmp_path)
    assert not report.ok
    st = reg.stats_for(yt)
    assert st["fail"] == 1 and st["ok"] == 0
    assert st["last_error_class"] == "network_failure", st
    assert reg.score(yt) < base_score, "failure did not degrade ranking"


def test_repeated_failures_sink_a_source_in_ordered_for(tmp_path):
    from nomorals.media.source_registry import (
        Licence, SourceDescriptor, SourceRegistry, TrustTier)

    reg = SourceRegistry()
    good = SourceDescriptor(
        nation="NG", language="en", domain="good.fm", name="GoodFM",
        kinds=("music",), licence=Licence.USER_UPLOAD,
        trust_tier=TrustTier.COMMUNITY)
    bad = SourceDescriptor(
        nation="NG", language="en", domain="bad.fm", name="BadFM",
        kinds=("music",), licence=Licence.USER_UPLOAD,
        trust_tier=TrustTier.COMMUNITY)
    reg.register(good)
    reg.register(bad)
    reg.record_outcome(good, ok=True)
    # below the stand-down threshold, failures visibly sink the source
    for _ in range(2):
        reg.record_outcome(bad, ok=False, error_class="network_failure")
    order = [d.domain for d in reg.ordered_for("NG", "en", "music")]
    assert order.index("good.fm") < order.index("bad.fm")
    assert reg.score(bad) < reg.score(good)
    # past the threshold (3+ consecutive), the source leaves the
    # active ordering entirely (SourceHealth stand-down semantics)
    for _ in range(4):
        reg.record_outcome(bad, ok=False, error_class="network_failure")
    st = reg.stats_for(bad)
    assert st["fail"] == 6 and st["last_error_class"] == "network_failure"
    order = [d.domain for d in reg.ordered_for("NG", "en", "music")]
    assert "bad.fm" not in order and "good.fm" in order


# ── (2) persistence ───────────────────────────────────────────────────


def test_registry_state_survives_restart_via_store(tmp_path):
    kv, _db = _kv(tmp_path)
    reg = _seeded(store=kv)
    yt = reg.for_url("https://www.youtube.com/watch?v=abc")
    reg.record_outcome(yt, ok=True, latency_s=1.0)
    reg.record_outcome(yt, ok=False, error_class="bot_block")
    assert reg.note_candidate("https://newtunes.example.com/x",
                              via="test") is True

    reg2 = _seeded(store=kv)  # a "restart": fresh registry, same store
    yt2 = reg2.for_url("https://www.youtube.com/watch?v=abc")
    st = reg2.stats_for(yt2)
    assert st["ok"] == 1 and st["fail"] == 1, st
    assert st["last_error_class"] == "bot_block", st
    cand_domains = [c.domain for c in reg2.candidates()]
    assert cand_domains == ["newtunes.example.com"], cand_domains


def test_wiping_persisted_state_returns_exact_seed_behaviour(tmp_path):
    kv, _db = _kv(tmp_path)
    from nomorals.media.source_registry import _PERSIST_KEY

    reg = _seeded(store=kv)
    seed_count = len(reg)
    reg.note_candidate("https://newtunes.example.com/x", via="test")
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] == 1
    assert len(reg) == seed_count + 1

    kv.delete(_PERSIST_KEY)
    reg2 = _seeded(store=kv)
    assert len(reg2) == seed_count
    assert reg2.candidates() == []
    assert reg2.for_url("https://newtunes.example.com/x") is None
    st = reg2.status()
    assert st["grown"] == 0 and st["candidates_pending"] == 0


def test_persisted_dead_source_does_not_resurrect_as_trusted(tmp_path):
    kv, _db = _kv(tmp_path)
    reg = _seeded(store=kv)
    assert reg.note_candidate("https://deadtunes.example.com/x",
                              via="test") is True
    for _ in range(3):
        reg.verify_health(fetcher=_dead_fetcher)
    # the failing candidate was shed, not promoted
    assert reg.for_url("https://deadtunes.example.com/x") is None

    reg2 = _seeded(store=kv)
    assert reg2.for_url("https://deadtunes.example.com/x") is None
    assert "deadtunes.example.com" in [
        s["domain"] for s in reg2.status()["shed_list"]]


# ── (4) discovery intake: candidates are untrusted until verified ─────


def test_candidate_is_untrusted_and_serves_nothing(tmp_path):
    reg = _seeded()
    assert reg.note_candidate("https://freshbeats.example.org/song",
                              via="download") is True
    # not a source: absent from the registry pool, the ordering and
    # the extractor view — an unverified candidate serves no downloads
    assert all(d.domain != "freshbeats.example.org" for d in reg.all())
    assert reg.for_url("https://freshbeats.example.org/song") is None
    assert "freshbeats.example.org" not in [
        d.domain for d in reg.ordered_for("", "", "music")]
    from nomorals.media.download_kit import ExtractorRegistry
    view = ExtractorRegistry(registry=reg)
    desc = view.best_for("https://freshbeats.example.org/song")
    assert desc is not None and desc.name == "ytdlp-generic"
    # duplicate intake is a no-op, known sources are never candidates
    assert reg.note_candidate("https://freshbeats.example.org/other",
                              via="download") is False
    assert reg.note_candidate("https://www.youtube.com/watch?v=1",
                              via="download") is False


def test_verified_candidate_is_promoted_by_the_health_loop(tmp_path):
    reg = _seeded()
    reg.note_candidate("https://freshbeats.example.org/song", via="test")
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] == 1, summary
    desc = reg.for_url("https://freshbeats.example.org/song")
    assert desc is not None and desc.domain == "freshbeats.example.org"
    assert "freshbeats.example.org" in [
        d.domain for d in reg.ordered_for("", "", "music")]
    st = reg.status()
    assert st["grown"] == 1 and st["candidates_pending"] == 0
    src = [s for s in st["sources"]
           if s["domain"] == "freshbeats.example.org"]
    assert src and src[0]["origin"] == "grown"
    assert src[0]["health"] == "verified-live"


def test_download_from_unknown_host_records_a_candidate(tmp_path):
    from nomorals.media.download_kit import ResumeReport
    from nomorals.media.downloader import MediaDownloader
    import hashlib

    reg = _seeded()
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg,
                         direct_retries=1)

    def fake_kit(self, url, dest, **kw):
        data = b"ID3" + b"x" * 64
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(data)
        return ResumeReport(ok=True, path=str(dest),
                            size_bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest())

    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("connection reset")), \
            mock.patch("nomorals.media.download_kit.ResumeDownloader.download",
                       fake_kit), \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        report = dl.download("https://cdn.example.test/track.mp3",
                             dest_dir=tmp_path)
    assert report.ok, report.summary()
    assert [c.domain for c in reg.candidates()] == ["cdn.example.test"]
    # the candidate did not become a trusted source by being downloaded
    assert reg.for_url("https://cdn.example.test/track.mp3") is None


# ── (3) health loop + scheduler job + tool binding ────────────────────


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


def test_source_health_job_registers_routine_and_idempotent(tmp_path):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.media.source_jobs import ensure_source_health_jobs

    ctx, _ = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = mock.MagicMock()
    out = ensure_source_health_jobs(sched)
    assert [o["name"] for o in out] == ["download source health"]
    rows = [j for j in sched.list_jobs()
            if j["name"] == "download source health"]
    assert len(rows) == 1
    assert rows[0]["spec"] == "every 21600s", rows[0]["spec"]
    payload = json.loads(ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE name = ?",
        ("download source health",))["payload"] or "{}")
    assert payload == {"tool": "download_sources",
                       "args": {"action": "health"}}, payload
    # routine (pass #71/#84 semantics): quiet bookkeeping — results
    # land in the registry status, not owner alerts — while a failed
    # run still records as failed.
    row = ctx.db.query_one(
        "SELECT routine FROM schedule_jobs WHERE name = ?",
        ("download source health",))
    assert int(row["routine"]) == 1, row
    ensure_source_health_jobs(sched)
    assert len([j for j in sched.list_jobs()
                if j["name"] == "download source health"]) == 1


def test_source_health_job_runs_through_the_real_scheduler(
        tmp_path, monkeypatch):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.media import source_registry as sr
    from nomorals.media.source_jobs import ensure_source_health_jobs

    monkeypatch.setattr(sr, "_default_health_fetch", _ok_fetcher)
    ctx, _ = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = mock.MagicMock()
    ensure_source_health_jobs(sched)
    row = [j for j in sched.list_jobs()
           if j["name"] == "download source health"][0]
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    runs = ctx.db.query(
        "SELECT ok FROM schedule_runs WHERE job_id = ?", (row["id"],))
    assert len(runs) == 1 and runs[0]["ok"] == 1, runs
    # the run really verified sources against the context-bound
    # registry: its status now carries a health verdict for seeds
    from nomorals.media.downloader import MediaDownloader
    st = MediaDownloader(ctx).source_registry().status()
    assert st["sources"], st
    assert all(s["health"] == "verified-live" for s in st["sources"]), st


def test_download_sources_tool_status_action(tmp_path):
    ctx, registry = _sched_ctx(tmp_path)
    outcome = registry.call("download_sources", action="status")
    assert outcome.ok, getattr(outcome, "error", outcome)
    data = outcome.data if hasattr(outcome, "data") else outcome
    # ToolRegistry.call returns an Outcome; the payload rides .value
    payload = getattr(outcome, "value", None) or data
    assert isinstance(payload, dict), payload
    assert payload["seeds"] >= 29, payload
    assert payload["candidates_pending"] == 0


# ── (5) status surface: /download status ──────────────────────────────


def test_download_status_surface_reports_registry_state(tmp_path):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    ctx, _ = _sched_ctx(tmp_path)
    obj = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    obj.context = ctx
    text = obj._control_download("status")
    assert "download sources" in text.lower(), text
    assert "seed" in text.lower(), text
    # and the plain usage path is untouched
    usage = obj._control_download("")
    assert usage.startswith("usage: /download"), usage
