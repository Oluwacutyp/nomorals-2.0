"""Wedge-defect pins (gapfix112, 2026-11-21).

The owner sent screenshots of Devon failing on EC2: the arena
researcher and the watch loop tick each wedged past the scheduler's
300s wall (Path.open / Condition.wait), the daily backup paged every
morning for a missing key, and /play took minutes to even get
processed.  Every pin below asserts the BOUNDED behaviour — a budget
inside the job, an ack before the search, a page once per streak —
never the wall itself.
"""
import base64
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# Module-level imports only: the NEW names (arena_tick_budget,
# survey deadline, /backup key, resolver budget, _play_resolve_and_send)
# do not exist on the pristine base, so each test resolves its names
# through its module — on pristine the test then fails individually
# (the failing-first proof) instead of breaking collection for all.
import nomorals.agents.arena_researcher.loop as arena_loop_mod
import nomorals.agents.arena_researcher.survey as arena_survey_mod
import nomorals.agents.notifier as notifier_mod
import nomorals.agents.ops_alerts as ops_mod
import nomorals.agents.reasoning as reasoning_mod
import nomorals.agents.watcher as watcher_mod
import nomorals.backup.coordinator as backup_coord_mod
import nomorals.backup.errors as backup_errors_mod
import nomorals.backup.report as backup_report_mod
import nomorals.media.resolver as resolver_mod
import nomorals.social.chat.control as control_mod

# ── 1. arena survey: deadline stops the walk, partial is honest ────

def _tree_with_files(tmp_path: Path, n: int = 60) -> Path:
    pkg = tmp_path / "nomorals"
    pkg.mkdir()
    for i in range(n):
        (pkg / f"mod_{i:02d}.py").write_text(
            "def work():\n    pass\n\n"
            "def silent():\n    try:\n        work()\n    except Exception:\n        pass\n",
            encoding="utf-8")
    return tmp_path


def test_survey_deadline_returns_partial_fast(tmp_path):
    root = _tree_with_files(tmp_path)
    t0 = time.monotonic()
    findings = arena_survey_mod.survey_repo(root, deadline_s=0.0001)
    took = time.monotonic() - t0
    assert findings.partial is True
    assert findings.surveyed_files < findings.total_files
    assert findings.total_files == 60
    assert took < 5.0
    # A partial survey is still a list of real findings.
    assert isinstance(findings, list)


def test_survey_no_deadline_covers_the_tree(tmp_path):
    root = _tree_with_files(tmp_path)
    findings = arena_survey_mod.survey_repo(root)
    assert findings.partial is False
    assert findings.surveyed_files == findings.total_files == 60


def test_arena_budgets_default_and_env(monkeypatch):
    monkeypatch.delenv("NM_ARENA_TICK_BUDGET_S", raising=False)
    monkeypatch.delenv("NM_ARENA_SURVEY_BUDGET_S", raising=False)
    assert arena_loop_mod.arena_tick_budget() == 240.0
    assert arena_loop_mod.arena_survey_budget() == 120.0
    monkeypatch.setenv("NM_ARENA_TICK_BUDGET_S", "200")
    monkeypatch.setenv("NM_ARENA_SURVEY_BUDGET_S", "30")
    assert arena_loop_mod.arena_tick_budget() == 200.0
    assert arena_loop_mod.arena_survey_budget() == 30.0
    # Clamped inside the scheduler's 300s wall, never above it.
    monkeypatch.setenv("NM_ARENA_TICK_BUDGET_S", "9999")
    assert arena_loop_mod.arena_tick_budget() == 290.0


def test_mine_sota_stops_on_tick_deadline():
    ArenaResearcher = arena_loop_mod.ArenaResearcher

    loop = ArenaResearcher.__new__(ArenaResearcher)
    loop.sota_organs = 3
    mined, count, claims = loop._mine_sota(
        ["one", "two", "three"], deadline=time.monotonic() - 1.0)
    assert (mined, count, claims) == (False, 0, {})


class _FakeSched:
    def __init__(self, rows):
        self.rows = list(rows)
        self.added = []
        self.policies = {}
        self.removed = []

    def list_jobs(self):
        return list(self.rows)

    def add(self, name, spec, kind, payload, **kw):
        # Simulates the real add()'s exact-replay dedup: a row that
        # already exists comes back carrying its OLD policy — the
        # caller must reconcile it, not trust the kwarg.
        row = {"id": "new", "name": name, "spec": spec,
               "overlap_policy": "concurrent"}
        self.added.append((name, spec, kw))
        return row

    def set_overlap_policy(self, job_id, policy):
        self.policies[job_id] = policy

    def remove(self, job_id):
        self.removed.append(job_id)


class _FakeScheduler:
    """Stand-in for nomorals.agents.scheduler.Scheduler.

    ``ensure_arena_researcher_job`` takes a CONTEXT and builds its own
    Scheduler, so the fake is installed as the Scheduler class and
    records its instances.  add() simulates exact-replay dedup: a row
    that already exists comes back carrying its OLD policy — the
    caller must reconcile it, not trust the kwarg.
    """

    instances: list = []

    def __init__(self, context):
        self.rows = list(getattr(context, "_rows", []))
        self.added = []
        self.policies = {}
        self.removed = []
        _FakeScheduler.instances.append(self)

    def list_jobs(self):
        return list(self.rows)

    def add(self, name, spec, kind, payload, **kw):
        row = {"id": "new", "name": name, "spec": spec,
               "overlap_policy": "concurrent"}
        self.added.append((name, spec, kw))
        return row

    def set_overlap_policy(self, job_id, policy):
        self.policies[job_id] = policy

    def remove(self, job_id):
        self.removed.append(job_id)


def _arena_context(rows):
    return SimpleNamespace(_rows=rows)


def test_arena_job_reconciles_persisted_concurrent_row(monkeypatch):
    _FakeScheduler.instances.clear()
    monkeypatch.setattr(
        "nomorals.agents.scheduler.Scheduler", _FakeScheduler)
    sched_ctx = _arena_context([{"id": "j1", "name": "arena researcher",
                                 "spec": "every 86400s",
                                 "overlap_policy": "concurrent"}])
    out = arena_loop_mod.ensure_arena_researcher_job(sched_ctx)
    assert out["already_scheduled"] is True
    # The persisted row is forced onto skip — add() alone cannot do
    # this for a row that already exists.
    assert _FakeScheduler.instances[0].policies == {"j1": "skip"}


def test_arena_job_registers_skip_on_fresh_row(monkeypatch):
    _FakeScheduler.instances.clear()
    monkeypatch.setattr(
        "nomorals.agents.scheduler.Scheduler", _FakeScheduler)
    sched_ctx = _arena_context([{"id": "j1", "name": "arena researcher",
                                 "spec": "every 12h",
                                 "overlap_policy": "concurrent"}])
    out = arena_loop_mod.ensure_arena_researcher_job(sched_ctx)
    assert out["scheduled"] is True
    sched = _FakeScheduler.instances[0]
    assert sched.removed == ["j1"]
    assert sched.added[0][2].get("overlap_policy") == "skip"
    assert sched.policies == {"new": "skip"}


# ── 2. watch loop tick: deadline stops starting new steps ──────────

_REPORT = {"sent": [], "self_healed": [], "cooldown": [],
           "findings": [], "pending": 2}


def _patch_watch(monkeypatch, ops_sleep=0.0):
    calls = {"redeliver": 0, "mine": 0}

    def _slow_ops(ctx, heal_background=True):
        if ops_sleep:
            time.sleep(ops_sleep)
        return dict(_REPORT)

    monkeypatch.setattr(ops_mod, "ops_alerts", _slow_ops)

    class _N:
        def __init__(self, ctx):
            pass

        def redeliver(self):
            calls["redeliver"] += 1
            return 0

    class _R:
        def __init__(self, ctx):
            pass

        def mine_preventions(self):
            calls["mine"] += 1
            return []

    monkeypatch.setattr(notifier_mod, "Notifier", _N)
    monkeypatch.setattr(reasoning_mod, "ReasoningAgent", _R)
    return calls


def test_watch_tick_deadline_skips_later_steps(monkeypatch):
    calls = _patch_watch(monkeypatch, ops_sleep=0.3)
    loop = watcher_mod.WatchLoop(SimpleNamespace(workspace=None))
    t0 = time.monotonic()
    report = loop.tick(deadline_s=0.2)
    took = time.monotonic() - t0
    # The first step (ops) still ran; nothing after it started.
    assert report["partial"] is True
    assert calls == {"redeliver": 0, "mine": 0}
    assert report["pending"] == 2
    assert took < 5.0


def test_watch_tick_full_budget_runs_every_step(monkeypatch):
    calls = _patch_watch(monkeypatch)
    loop = watcher_mod.WatchLoop(SimpleNamespace(workspace=None))
    report = loop.tick(deadline_s=60.0)
    assert report["partial"] is False
    assert calls == {"redeliver": 1, "mine": 1}
    assert report["tick"] == 1


def test_watch_job_registers_skip(monkeypatch):
    from nomorals.agents.watcher import ensure_watch_loop_job

    sched = _FakeSched([])
    job = ensure_watch_loop_job(sched)
    assert job is not None
    assert sched.added[0][2].get("overlap_policy") == "skip"
    assert sched.added[0][2].get("routine") is True


def test_relay_and_triage_jobs_register_skip():
    from nomorals.relay.jobs import ensure_ticket_relay_job
    from nomorals.triage.jobs import ensure_triage_job

    for ensure in (ensure_ticket_relay_job, ensure_triage_job):
        sched = _FakeSched([])
        job = ensure(sched)
        assert job is not None
        assert sched.added[0][2].get("overlap_policy") == "skip"


# ── 3. backup: key first, page once per streak ─────────────────────

class _KeylessVault:
    def get(self, *args, **kwargs):
        raise RuntimeError("no such credential")


def test_backup_key_checked_before_snapshot(tmp_path, monkeypatch):
    calls = []

    def _take_snapshot(root, work_dir, label=""):
        calls.append(label)
        raise AssertionError("snapshot must not run before the key check")

    monkeypatch.setattr(
        "nomorals.backup.coordinator.take_snapshot", _take_snapshot)
    coord = backup_coord_mod.BackupCoordinator(root=tmp_path, work_dir=tmp_path,
                              vault=_KeylessVault())
    with pytest.raises(backup_errors_mod.BackupKeyError):
        coord.run_backup(label="scheduled")
    assert calls == []


class _Publisher:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body, **kw):
        self.calls.append((kind, title))
        return {"delivered": True}


def test_backup_failure_pages_once_per_streak(tmp_path):
    pub = _Publisher()
    err = RuntimeError("boom")
    first = backup_report_mod.report_backup_failure(pub, "backup-x", err,
                                  state_dir=tmp_path)
    assert first.get("paged", True) is not False
    assert len(pub.calls) == 1
    second = backup_report_mod.report_backup_failure(pub, "backup-y", RuntimeError("boom"),
                                   state_dir=tmp_path)
    assert second.get("paged") is False
    assert second.get("reason") == "same-streak"
    assert len(pub.calls) == 1
    # A different error class is a new problem — it pages again.
    third = backup_report_mod.report_backup_failure(pub, "backup-z", ValueError("other"),
                                  state_dir=tmp_path)
    assert third.get("paged", True) is not False
    assert len(pub.calls) == 2
    # A success clears the streak: the next failure pages again.
    backup_report_mod.clear_backup_failure_streak(tmp_path)
    fourth = backup_report_mod.report_backup_failure(pub, "backup-w", RuntimeError("boom"),
                                   state_dir=tmp_path)
    assert fourth.get("paged", True) is not False
    assert len(pub.calls) == 3


def test_backup_failure_without_state_dir_pages_every_call(tmp_path):
    pub = _Publisher()
    backup_report_mod.report_backup_failure(pub, "backup-x", RuntimeError("boom"))
    backup_report_mod.report_backup_failure(pub, "backup-y", RuntimeError("boom"))
    assert len(pub.calls) == 2


# ── 3b. scheduler: a failing job pages once per streak ─────────────

class _NullDB:
    def query(self, *args, **kwargs):
        raise RuntimeError("db down")

    def query_one(self, *args, **kwargs):
        return None

    def execute(self, *args, **kwargs):
        return None

    def transaction(self):
        import contextlib
        return contextlib.nullcontext()


def _sched_with_failing_tool(publish_calls):
    from nomorals.agents.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.db = _NullDB()
    s.context = SimpleNamespace()
    s.notifier = SimpleNamespace(
        publish=lambda *a, **k: publish_calls.append((a, k)))
    s._lock = threading.Lock()
    s._active = set()
    s._deferrals = {}
    s._tick_errors = 0
    s._last_tick_at = None
    s._started_at = None
    s._thread = None
    s.tick_seconds = 20.0
    s.redeliver_interval = 120.0
    s._last_redeliver_at = 1e18
    s._running_jobs = 0
    s.wall_seconds = 300.0
    import random
    s._rng = random.Random(0)
    s._run_payload_guarded = lambda row, t: (
        False,
        "job failed: tool __ticket_relay_sync__ failed: "
        "github is not connected",
        False, None, None, None)
    s._next_after_run = lambda row, now: (None, True)
    s._record_run = lambda *a, **k: None
    s._ledger = lambda *a, **k: None
    s._emit_bus = lambda *a, **k: None
    s._is_routine = lambda row: False
    return s


def test_scheduler_pages_failing_job_once_per_streak():
    calls = []
    s = _sched_with_failing_tool(calls)
    row = {"id": "j1", "name": "ticket relay sync",
           "payload_kind": "tool", "last_result": ""}
    out1 = s._execute(dict(row))
    assert out1["ok"] is False
    assert len(calls) == 1
    # Second run, still failing — the persisted last_result says so.
    row2 = dict(row)
    row2["last_result"] = (
        "job failed: tool __ticket_relay_sync__ failed: "
        "github is not connected")
    out2 = s._execute(row2)
    assert out2["ok"] is False
    assert len(calls) == 1


# ── 4. /backup key: generate once, refuse a second, show stored ───

class _MemoryVault:
    def __init__(self):
        self.stored = None

    def get(self, service, username, mark_used=False):
        if self.stored is None:
            raise RuntimeError("no such credential")
        return SimpleNamespace(password=self.stored)

    def store(self, service, username, value, credential_type=None,
              tags=None, metadata=None):
        self.stored = value


def _backup_self(tmp_path):
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    self_obj = RuntimeReachMixin.__new__(RuntimeReachMixin)
    self_obj.context = SimpleNamespace(
        db=object(),
        settings=SimpleNamespace(backup_dir=tmp_path,
                                 db_path=tmp_path / "x.db"))
    return self_obj


def test_backup_key_generate_refuse_show(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    vault = _MemoryVault()
    monkeypatch.setattr(
        "nomorals.accounts.vault.open_vault",
        lambda db, context=None, actor=None: vault)
    self_obj = _backup_self(tmp_path)
    out = RuntimeReachMixin._control_backup(self_obj, "key")
    assert "shown ONCE" in out
    assert vault.stored is not None
    # The printed key IS the stored key (base64 round-trip).
    assert vault.stored in out
    assert base64.b64decode(vault.stored) and len(
        base64.b64decode(vault.stored)) == 32
    # A second generate refuses — rotating would orphan generations.
    out2 = RuntimeReachMixin._control_backup(self_obj, "key")
    assert "already stored" in out2
    assert "refusing" in out2
    # Show prints the stored key for the offline copy.
    out3 = RuntimeReachMixin._control_backup(self_obj, "key show")
    assert vault.stored in out3


def test_backup_key_show_without_key(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    vault = _MemoryVault()
    monkeypatch.setattr(
        "nomorals.accounts.vault.open_vault",
        lambda db, context=None, actor=None: vault)
    self_obj = _backup_self(tmp_path)
    out = RuntimeReachMixin._control_backup(self_obj, "key show")
    assert "no backup key" in out
    assert "/backup key" in out


def test_backup_arity_allows_key_show():
    assert control_mod.CONTROL_COMMANDS["backup"] == (0, 2)


# ── 5. resolver: chain budget stops starting new sources ──────────

def test_run_chain_stops_on_deadline():
    resolver = resolver_mod.SourceResolver(SimpleNamespace())
    seen = []

    def slow1():
        seen.append("slow1")
        time.sleep(0.3)
        return resolver_mod.ResolvedAudio(ok=False)

    def slow2():
        seen.append("slow2")
        return resolver_mod.ResolvedAudio(ok=False)

    result = resolver_mod.ResolvedAudio()
    out = resolver._run_chain(
        result, [("slow1", slow1), ("slow2", slow2)],
        deadline=time.monotonic() + 0.2)
    assert seen == ["slow1"]
    assert out.truncated is True
    assert any("chain budget" in a for a in out.attempts)


def test_sweep_stops_on_deadline():
    resolver = resolver_mod.SourceResolver(SimpleNamespace())
    called = []

    def slow_mp3pm(q, limit):
        time.sleep(0.3)
        return []

    def rec_juice(q, limit):
        called.append("juice")
        return []

    resolver._mp3pm_search_many = slow_mp3pm
    resolver._mp3juice_search_many = rec_juice
    _cands, attempts = resolver.search_candidates_detailed(
        "some song", budget_s=0.2)
    assert called == []
    assert any("chain budget" in a for a in attempts)


def test_resolve_threads_deadline_into_chain(tmp_path, monkeypatch):
    resolver = resolver_mod.SourceResolver(SimpleNamespace())
    seen = {}
    orig = resolver_mod.SourceResolver._run_chain

    def spy(self, result, strategies, *, deadline=None):
        seen["deadline"] = deadline
        return orig(self, result, strategies, deadline=deadline)

    monkeypatch.setattr(resolver_mod.SourceResolver, "_run_chain", spy)
    song = tmp_path / "song.mp3"
    song.write_bytes(b"fake audio")
    before = time.monotonic()
    out = resolver.resolve(str(song))
    assert out.ok is True
    assert seen["deadline"] is not None
    assert before + 50.0 < seen["deadline"] < before + 70.0


# ── 6. /play: resolve + sweep run behind the ack ───────────────────

class _FakeEngine:
    def __init__(self):
        self.enqueued = []

    def _enqueue(self, path, kind, title, artist="", duration=0.0):
        item = {"path": path, "kind": kind, "title": title,
                "artist": artist, "duration": duration,
                "index": len(self.enqueued)}
        self.enqueued.append(item)
        return item

    def queue(self):
        return list(self.enqueued)

    def remove(self, idx):
        self.enqueued = [e for e in self.enqueued
                         if e.get("index") != idx]
        return "removed"


def _media_self():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    self_obj = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    self_obj.context = SimpleNamespace()
    return self_obj


def test_play_resolve_and_send_failure_returns_honest_line(monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    class _FailResolver:
        def __init__(self, ctx):
            pass

        def resolve(self, q):
            return resolver_mod.ResolvedAudio(
                ok=False, attempts=["mp3pm-search — no results"],
                errors=["mp3pm-search: no results"],
                hint="try another title")

        def search_candidates_detailed(self, q):
            raise AssertionError("sweep must not run on resolve failure")

    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver", _FailResolver)
    self_obj = _media_self()
    text, sent = RuntimeMediaMixin._play_resolve_and_send(
        self_obj, _FakeEngine(), "some song", "some song",
        "some song", "", SimpleNamespace())
    assert sent is False
    assert "couldn't find" in text


def test_play_resolve_and_send_success_uses_fallthrough(monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    hit = resolver_mod.ResolvedAudio(
        ok=True, path_or_url="https://x.example/track.mp3",
        kind="youtube", title="Real Title", artist="Real Artist",
        duration=180.0, source_name="youtube-search")

    class _OkResolver:
        def __init__(self, ctx):
            pass

        def resolve(self, q):
            return hit

        def search_candidates_detailed(self, q):
            return [hit], ["youtube-search — ok (1)"]

    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver", _OkResolver)
    seen = {}

    def _fake_fallthrough(self, engine, first_item, fallbacks, merged,
                          chat, match=None, query=""):
        seen["first_item"] = first_item
        seen["fallbacks"] = fallbacks
        return "delivered: Real Title", True

    monkeypatch.setattr(RuntimeMediaMixin, "_play_send_with_fallthrough",
                        _fake_fallthrough)
    engine = _FakeEngine()
    self_obj = _media_self()
    text, sent = RuntimeMediaMixin._play_resolve_and_send(
        self_obj, engine, "some song", "some song",
        "some song", "", SimpleNamespace())
    assert (text, sent) == ("delivered: Real Title", True)
    assert seen["first_item"]["title"] == "Real Title"
    assert engine.enqueued and engine.enqueued[0]["title"] == "Real Title"


def test_play_resolve_and_send_preview_refuses(monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    hit = resolver_mod.ResolvedAudio(
        ok=True, path_or_url="https://x.example/preview.m4a",
        kind="itunes", title="Preview Track", artist="",
        duration=30.0, source_name="itunes-search",
        downloadable=False, preview_only=True)

    class _PreviewResolver:
        def __init__(self, ctx):
            pass

        def resolve(self, q):
            return hit

        def search_candidates_detailed(self, q):
            return [], ["itunes-search — ok (1)"]

    monkeypatch.setattr(
        "nomorals.media.resolver.SourceResolver", _PreviewResolver)
    monkeypatch.setattr(
        RuntimeMediaMixin, "_play_send_with_fallthrough",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("preview must never reach the fallthrough")))
    engine = _FakeEngine()
    self_obj = _media_self()
    text, sent = RuntimeMediaMixin._play_resolve_and_send(
        self_obj, engine, "some song", "some song",
        "some song", "", SimpleNamespace())
    assert sent is False
    assert "preview" in text.lower()


def test_play_job_relabels_to_resolved_title():
    """The ack-first job is registered under the raw query; once the
    worker resolves, its title label must become the resolved title —
    the duplicate guard, /download cancel and the harvester all look
    jobs up by that label (2026-11-21 wedge fix-up)."""
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    from nomorals.core.jobs import shared_registry

    self_obj = _media_self()
    # No-op outside a job: never raises, never relabels anything.
    RuntimeMediaMixin._play_dl_relabel_current_job(
        self_obj, "No Job Here")

    job = shared_registry().submit(
        lambda: RuntimeMediaMixin._play_dl_relabel_current_job(
            self_obj, "Resolved Title"),
        tool="wedge-relabel-pin",
        labels={"owner": "play", "chat": "telegram:1",
                "title": "raw query"})
    done = shared_registry().collect(job.id, timeout=10.0)
    assert done is not None and done.done()
    assert done.snapshot()["labels"]["title"] == "Resolved Title"
