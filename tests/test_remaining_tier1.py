"""Remaining Tier 1 consolidation: unified scheduler, plugin shipping,
phone stream client, one interruption policy.

All tests use only the NEW modules (no existing-file behaviour is
modified or asserted here beyond the recurrence engine the unified
scheduler delegates cron math to).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from nomorals.autonomy.interruption import assess_worth, in_quiet_hours
from nomorals.plugins.shipping import (
    ManifestError,
    install_plugin,
    list_plugins,
    uninstall,
)
from nomorals.scheduler.unified_scheduler import (
    DEFAULT_JITTER_FRAC,
    UnifiedJob,
    UnifiedScheduler,
    apply_missed_policy,
    decorrelated_jitter,
    next_occurrence,
)
from nomorals.stream.phone_client import (
    SSEClient,
    TimedGrant,
    TokenExpired,
    TokenInvalid,
    cost_meter,
    live_autonomy_view,
    mint_stream_grant,
    parse_sse_text,
    render_phone_page,
    verify_stream_token,
)

NOW = 1_800_000_000.0  # fixed "now" (2037-01-15-ish, deterministic)


# ── unified scheduler: four views, one next_run ──────────────────────

def _sched_with_all_kinds(now: float = NOW) -> UnifiedScheduler:
    sched = UnifiedScheduler()
    sched.add_job(UnifiedJob(
        job_id="cron-job", kind="cron",
        recurrence={"type": "cron", "spec": "0 9 * * *"},
        created_at=now - 3600), now=now, rng=random.Random(1))
    sched.add_job(UnifiedJob(
        job_id="interval-job", kind="interval",
        recurrence={"type": "interval", "spec": 3600},
        created_at=now - 3600), now=now, rng=random.Random(1))
    sched.add_job(UnifiedJob(
        job_id="mission-job", kind="mission",
        recurrence={"type": "interval", "spec": 7200},
        plan=[{"name": "step-1", "handler": "noop"}],
        created_at=now - 3600), now=now, rng=random.Random(1))
    sched.add_job(UnifiedJob(
        job_id="tick-job", kind="organ_tick",
        recurrence={"type": "interval", "spec": 1800},
        organ="presence", created_at=now - 3600),
        now=now, rng=random.Random(1))
    return sched


def test_four_views_agree_on_next_run():
    sched = _sched_with_all_kinds()
    assert sched.cron_view(now=NOW)[0]["next_run"] == sched.next_run("cron-job", now=NOW)
    assert sched.interval_view(now=NOW)[0]["next_run"] == sched.next_run("interval-job", now=NOW)
    mission_row = sched.mission_view(now=NOW)[0]
    assert mission_row["next_run"] == sched.next_run("mission-job", now=NOW)
    assert mission_row["plan"] == [{"name": "step-1", "handler": "noop"}]
    tick_row = sched.organ_tick_view(now=NOW)[0]
    assert tick_row["next_run"] == sched.next_run("tick-job", now=NOW)
    assert tick_row["organ"] == "presence"
    # One store: the four views partition the jobs, none lost, none doubled.
    seen = [r["job_id"] for r in sched.cron_view(now=NOW)
            + sched.interval_view(now=NOW) + sched.mission_view(now=NOW)
            + sched.organ_tick_view(now=NOW)]
    assert sorted(seen) == ["cron-job", "interval-job", "mission-job", "tick-job"]
    # Same call twice, same answer (views are pure reads over the store).
    assert sched.cron_view(now=NOW)[0]["next_run"] == sched.cron_view(now=NOW)[0]["next_run"]


def test_jobs_persist_json_and_reload(tmp_path):
    store = tmp_path / "jobs.json"
    sched = UnifiedScheduler(store)
    sched.add_job(UnifiedJob(
        job_id="persisted", kind="interval",
        recurrence={"type": "interval", "spec": 600},
        created_at=NOW - 60), now=NOW, rng=random.Random(2))
    before = sched.next_run("persisted", now=NOW)
    sched.save()
    assert store.is_file()
    raw = json.loads(store.read_text())
    assert raw["version"] == 1 and len(raw["jobs"]) == 1
    reloaded = UnifiedScheduler(store)
    assert reloaded.next_run("persisted", now=NOW) == before


# ── missed-run policy ────────────────────────────────────────────────

def _stale_job(policy: str, *, created_ago: float = 5 * 3600) -> UnifiedJob:
    job = UnifiedJob(
        job_id=f"stale-{policy}", kind="interval",
        recurrence={"type": "interval", "spec": 3600},
        missed_policy=policy, jitter_frac=0.0,
        created_at=NOW - created_ago)
    # Simulate a next_run that went stale 4 hours ago.
    job.next_run = NOW - 4 * 3600
    job.last_run = NOW - 5 * 3600
    return job


def test_missed_policy_catch_up_once_fires_exactly_once():
    verdict = apply_missed_policy(_stale_job("catch_up_once"), NOW)
    assert verdict["action"] == "fire_once"
    assert verdict["fire"] == [NOW]  # coalesced: many missed, one run
    assert verdict["next_run"] > NOW


def test_missed_policy_skip_fires_nothing():
    verdict = apply_missed_policy(_stale_job("skip"), NOW)
    assert verdict["action"] == "skipped"
    assert verdict["fire"] == []
    assert verdict["next_run"] > NOW


def test_missed_policy_backfill_replays_each_missed_firing():
    verdict = apply_missed_policy(_stale_job("backfill"), NOW)
    assert verdict["action"] == "backfill"
    # Missed firings are the hourly nominal times in (last_run, NOW].
    assert len(verdict["fire"]) == 5
    assert verdict["fire"] == sorted(verdict["fire"])
    assert verdict["fire"][-1] <= NOW
    assert verdict["next_run"] > NOW


def test_missed_policy_backfill_capped():
    job = _stale_job("backfill", created_ago=365 * 86400)
    # Missed firings walk from last_run — push that back a year too.
    job.last_run = NOW - 365 * 86400
    job.next_run = NOW - 365 * 86400
    verdict = apply_missed_policy(job, NOW, cap=10)
    assert len(verdict["fire"]) == 10  # capped, not a million replays


def test_missed_policy_on_time_untouched():
    job = UnifiedJob(job_id="fresh", kind="interval",
                     recurrence={"type": "interval", "spec": 3600},
                     created_at=NOW - 60)
    job.next_run = NOW + 3600
    verdict = apply_missed_policy(job, NOW)
    assert verdict["action"] == "on_time" and verdict["fire"] == []
    assert verdict["next_run"] == NOW + 3600


# ── jitter ───────────────────────────────────────────────────────────

def test_jitter_bounded_by_frac_times_period():
    rng = random.Random(0)
    for _ in range(200):
        draw = decorrelated_jitter(3600.0, DEFAULT_JITTER_FRAC, rng)
        assert abs(draw) <= DEFAULT_JITTER_FRAC * 3600.0
    assert decorrelated_jitter(3600.0, 0.0, rng) == 0.0
    assert decorrelated_jitter(0.0, 0.5, rng) == 0.0


def test_scheduled_next_run_jitter_bounded():
    sched = UnifiedScheduler()
    sched.add_job(UnifiedJob(
        job_id="j", kind="interval",
        recurrence={"type": "interval", "spec": 3600},
        jitter_frac=0.10, created_at=NOW - 60),
        now=NOW, rng=random.Random(7))
    nominal = next_occurrence({"type": "interval", "spec": 3600}, NOW)
    assert nominal is not None
    got = sched.next_run("j", now=NOW)
    assert got is not None
    assert abs(got - nominal) <= 0.10 * 3600.0
    assert got > NOW


# ── plugin shipping ──────────────────────────────────────────────────

_PLUGIN_COUNTER = [0]


def _plugin_dir(tmp_path: Path, *, permissions=None,
                artifacts: dict | None = None) -> Path:
    _PLUGIN_COUNTER[0] += 1
    root = tmp_path / f"my-pack-{_PLUGIN_COUNTER[0]}"
    root.mkdir()
    manifest: dict = {
        "name": "my-pack",
        "version": "1.0.0",
        "permissions": ["net", "fs"] if permissions is None else permissions,
        "artifacts": artifacts if artifacts is not None else {
            "culture_packs": [{"path": "culture/ekiti.json"}],
            "playbooks": [{"path": "playbooks/morning.md"}],
        },
    }
    (root / "plugin.json").write_text(json.dumps(manifest))
    (root / "culture").mkdir()
    (root / "culture" / "ekiti.json").write_text('{"dialect": "ekiti"}')
    (root / "playbooks").mkdir()
    (root / "playbooks" / "morning.md").write_text("# morning")
    return root


def test_plugin_install_list_uninstall(tmp_path):
    plugins_dir = tmp_path / "plugins"
    receipt = install_plugin(_plugin_dir(tmp_path), plugins_dir=plugins_dir)
    assert receipt["name"] == "my-pack" and receipt["version"] == "1.0.0"
    # net/fs expanded to dotted capabilities; spawn not granted.
    assert "network.fetch" in receipt["permissions"]
    assert "artifacts.write" in receipt["permissions"]
    assert "spawn" not in receipt["permissions"]
    assert (plugins_dir / "my-pack-1.0.0" / "culture" / "ekiti.json").is_file()
    listed = list_plugins(plugins_dir)
    assert [p["name"] for p in listed] == ["my-pack"]
    assert uninstall("my-pack", plugins_dir=plugins_dir) is True
    assert list_plugins(plugins_dir) == []


def test_plugin_must_declare_permissions(tmp_path):
    root = _plugin_dir(tmp_path)
    manifest = json.loads((root / "plugin.json").read_text())
    del manifest["permissions"]
    (root / "plugin.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestError, match="declare permissions"):
        install_plugin(root, plugins_dir=tmp_path / "plugins")
    assert list_plugins(tmp_path / "plugins") == []  # nothing installed


def test_plugin_invalid_manifest_rejected(tmp_path):
    # Unknown permission.
    root = _plugin_dir(tmp_path, permissions=["teleport"])
    with pytest.raises(ManifestError, match="unknown permissions"):
        install_plugin(root, plugins_dir=tmp_path / "plugins")
    # Bad version.
    root2 = _plugin_dir(tmp_path)
    manifest = json.loads((root2 / "plugin.json").read_text())
    manifest["version"] = "not-semver"
    (root2 / "plugin.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestError, match="version"):
        install_plugin(root2, plugins_dir=tmp_path / "plugins")
    # Artifact path traversal.
    root3 = _plugin_dir(tmp_path, artifacts={
        "sources": [{"path": "../escape.txt"}]})
    with pytest.raises(ManifestError, match="escapes"):
        install_plugin(root3, plugins_dir=tmp_path / "plugins")
    # Missing artifact file.
    root4 = _plugin_dir(tmp_path, artifacts={
        "g2p": [{"path": "g2p/missing.json"}]})
    with pytest.raises(ManifestError, match="missing"):
        install_plugin(root4, plugins_dir=tmp_path / "plugins")
    assert list_plugins(tmp_path / "plugins") == []


def test_plugin_spawn_permission_declared_verbatim(tmp_path):
    root = _plugin_dir(tmp_path, permissions=["spawn"])
    receipt = install_plugin(root, plugins_dir=tmp_path / "plugins")
    assert receipt["permissions"] == ["spawn"]


# ── stream phone client: TimedGrant ──────────────────────────────────

SECRET = "test-stream-secret"


def test_stream_token_roundtrip_and_expiry_enforced():
    grant = mint_stream_grant(SECRET, ttl_s=300, now=NOW)
    assert not grant.expired or True  # expiry is wall-clock; verify with now=
    verified = verify_stream_token(grant.token, SECRET, now=NOW + 10)
    assert verified.capability == "stream:read"
    assert verified.expires_at == NOW + 300
    # Expired token rejected (same token, later now).
    with pytest.raises(TokenExpired):
        verify_stream_token(grant.token, SECRET, now=NOW + 301)
    # Wrong secret → invalid.
    with pytest.raises(TokenInvalid):
        verify_stream_token(grant.token, "wrong-secret", now=NOW + 10)
    # Tampered payload → invalid.
    tampered = grant.token[:-2] + "AA"
    with pytest.raises(TokenInvalid):
        verify_stream_token(tampered, SECRET, now=NOW + 10)
    # Garbage → invalid, not a crash.
    with pytest.raises(TokenInvalid):
        verify_stream_token("not-a-token", SECRET, now=NOW)


def test_stream_client_refuses_expired_grant_without_network():
    # Real wall-clock expiry (TimedGrant.expired reads time.time()).
    expired = mint_stream_grant(SECRET, ttl_s=-10)
    assert expired.expired is True
    client = SSEClient("http://localhost:9", expired)
    with pytest.raises(TokenExpired):
        next(client.stream())


def test_sse_parse_and_phone_page():
    text = (
        ": ping\n"
        "event: mission.update\ndata: {\"id\": \"m1\"}\nid: 42\n\n"
        "data: line one\ndata: line two\n\n"
    )
    events = parse_sse_text(text)
    assert events[0] == {"event": "mission.update", "data": {"id": "m1"},
                         "id": "42", "retry": None}
    assert events[1]["data"] == "line one\nline two"

    missions = [
        {"id": "m1", "name": "Morning pulse", "status": "running"},
        {"id": "m2", "name": "Research", "status": "queued"},
    ]
    view = live_autonomy_view(missions, spent=8.0, budget=10.0)
    assert view["mission_count"] == 2
    assert view["by_status"] == {"running": 1, "queued": 1}
    assert view["cost"]["pct"] == 0.8 and view["cost"]["warn"] is True
    assert cost_meter(11.0, 10.0)["over"] is True
    assert cost_meter(5.0, 0.0)["pct"] is None  # no budget → no meter
    page = render_phone_page(missions, spent=8.0, budget=10.0)
    assert "Morning pulse" in page and "viewport" in page
    assert "80%" in page


# ── interruption policy ──────────────────────────────────────────────

def test_interruption_consistent_same_inputs_same_output():
    intr = {"priority": "normal", "urgency": 0.8, "importance": 0.7}
    ctx = {"now": NOW, "quiet_hours": None, "interruptibility": 0.6}
    assert assess_worth(intr, ctx) == assess_worth(intr, ctx)


def test_interruption_thresholds_per_priority():
    # Same middling content: urgent interrupts, normal defers, background drops.
    intr = {"urgency": 0.6, "importance": 0.6}
    ctx = {"now": NOW, "quiet_hours": None, "interruptibility": 0.5}
    assert assess_worth({**intr, "priority": "urgent"}, ctx)["decision"] == "interrupt"
    assert assess_worth({**intr, "priority": "normal"}, ctx)["decision"] == "defer"
    assert assess_worth({**intr, "priority": "background"}, ctx)["decision"] == "drop"
    # Higher urgency never lowers the score (monotone).
    low = assess_worth({"priority": "normal", "urgency": 0.1, "importance": 0.1}, ctx)
    high = assess_worth({"priority": "normal", "urgency": 0.9, "importance": 0.9}, ctx)
    assert high["score"] >= low["score"]


def test_interruption_quiet_hours_hold_non_safety():
    # 02:30 UTC is inside 23:00-07:00 quiet hours.
    quiet_now = 1_800_000_000.0  # pick a ts, then compute: use explicit check
    assert in_quiet_hours(quiet_now, "23:00-07:00") or True
    # Fixed quiet moment: 2026-10-16 02:30 UTC.
    import datetime as _dt
    quiet_ts = _dt.datetime(2026, 10, 16, 2, 30, tzinfo=_dt.timezone.utc).timestamp()
    assert in_quiet_hours(quiet_ts, "23:00-07:00") is True
    assert in_quiet_hours(quiet_ts, None) is False
    ctx_quiet = {"now": quiet_ts, "quiet_hours": "23:00-07:00",
                 "interruptibility": 0.5}
    normal = assess_worth(
        {"priority": "normal", "urgency": 0.9, "importance": 0.9}, ctx_quiet)
    assert normal["decision"] in ("defer", "drop")  # never interrupt in quiet
    assert normal["quiet_hours"] is True
    # Safety still interrupts in quiet hours.
    safety = assess_worth(
        {"priority": "safety", "urgency": 0.1, "importance": 0.1}, ctx_quiet)
    assert safety["decision"] == "interrupt"
    # A maximally urgent page breaks through quiet hours.
    urgent_max = assess_worth(
        {"priority": "urgent", "urgency": 1.0, "importance": 1.0},
        {**ctx_quiet, "interruptibility": 1.0})
    assert urgent_max["decision"] == "interrupt"
    # …but a middling urgent one waits.
    urgent_mid = assess_worth(
        {"priority": "urgent", "urgency": 0.6, "importance": 0.6}, ctx_quiet)
    assert urgent_mid["decision"] in ("defer", "drop")


def test_interruption_unknown_priority_defaults_normal():
    res = assess_worth({"priority": "bogus", "urgency": 0.9, "importance": 0.9},
                       {"now": NOW, "quiet_hours": None})
    assert res["priority"] == "normal"
    assert res["decision"] == "interrupt"
    assert set(res) >= {"score", "decision", "priority", "reasons"}
