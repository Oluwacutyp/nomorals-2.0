"""Wedge real fix (gapfix115): the survey completes, the tick completes,
and a slow job does not hold up the rest.

Falling-first: on the base (07f4602d) the survey has no cache (a full
survey is 37-47s here, 3-8x that on EC2 — past the 120s budget), the
watch tick has no write bound (one wedged writer stalls it 30s per
write toward the 300s wall), and IdempotencyStore issues schema writes
on every construction (so the ops scan dies before reading one
mission). The tests below fail on the base and pass on the fix.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.arena_researcher import survey as survey_mod
from nomorals.agents.arena_researcher.survey import survey_repo
from nomorals.core.errors import StorageError
from nomorals.storage.db import Database

# ── fixtures ──────────────────────────────────────────────────────────────

def _write_repo(root: Path) -> None:
    """A tiny repo with one live module, one imported module, and one
    dead module carrying a stub + a silent swallow."""
    pkg = root / "nomorals"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "used.py").write_text(
        "import nomorals.helper\n\n\ndef used():\n    return nomorals.helper.helper()\n",
        encoding="utf-8")
    (pkg / "helper.py").write_text(
        "def helper():\n    return 2\n", encoding="utf-8")
    (pkg / "dead.py").write_text(
        "def dead_stub():\n    pass\n\n\ndef noisy():\n    try:\n        raise ValueError()\n    except ValueError:\n        pass\n",
        encoding="utf-8")


def _finding_keys(findings) -> list[tuple[str, str, int]]:
    return sorted((f["id"], f["path"], f["line"]) for f in findings)


class _CountingParse:
    """Wraps ast.parse, counting calls (proves what was re-parsed)."""

    def __init__(self, real_parse):
        self.real_parse = real_parse
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.real_parse(*args, **kwargs)


# ── 1. survey cache ───────────────────────────────────────────────────────

class TestSurveyCache:
    def test_cold_cache_same_findings_as_full(self, tmp_path):
        _write_repo(tmp_path)
        cache = tmp_path / "cache.json"
        full = survey_repo(tmp_path)
        cold = survey_repo(tmp_path, cache_path=cache)
        assert _finding_keys(cold) == _finding_keys(full)
        assert cold.surveyed_files == cold.total_files
        assert cold.partial is False
        assert cache.exists()

    def test_warm_cache_parses_nothing_and_matches(self, tmp_path,
                                                   monkeypatch):
        _write_repo(tmp_path)
        cache = tmp_path / "cache.json"
        survey_repo(tmp_path, cache_path=cache)
        counter = _CountingParse(survey_mod.ast.parse)
        monkeypatch.setattr(survey_mod.ast, "parse", counter)
        warm = survey_repo(tmp_path, cache_path=cache)
        assert counter.calls == 0
        assert _finding_keys(warm) == _finding_keys(
            survey_repo(tmp_path, cache_path=cache))
        assert warm.surveyed_files == warm.total_files
        assert warm.partial is False

    def test_changed_file_reparsed_and_findings_follow(self, tmp_path,
                                                       monkeypatch):
        _write_repo(tmp_path)
        cache = tmp_path / "cache.json"
        survey_repo(tmp_path, cache_path=cache)
        dead = tmp_path / "nomorals" / "dead.py"
        dead.write_text(dead.read_text(encoding="utf-8")
                        + "\n\ndef another_stub():\n    pass\n",
                        encoding="utf-8")
        counter = _CountingParse(survey_mod.ast.parse)
        monkeypatch.setattr(survey_mod.ast, "parse", counter)
        touched = survey_repo(tmp_path, cache_path=cache)
        assert counter.calls == 1
        assert any(f["path"].endswith("dead.py")
                   and f["kind"] == "stub"
                   and f["line"] > 10 for f in touched)
        assert _finding_keys(touched) == _finding_keys(
            survey_repo(tmp_path))

    def test_corrupt_cache_ignored(self, tmp_path):
        _write_repo(tmp_path)
        cache = tmp_path / "cache.json"
        cache.write_text("this is not json", encoding="utf-8")
        found = survey_repo(tmp_path, cache_path=cache)
        assert _finding_keys(found) == _finding_keys(survey_repo(tmp_path))

    def test_wrong_root_cache_ignored(self, tmp_path, monkeypatch):
        _write_repo(tmp_path)
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps(
            {"version": survey_mod.CACHE_VERSION,
             "root": "/some/other/repo", "files": {}}),
            encoding="utf-8")
        counter = _CountingParse(survey_mod.ast.parse)
        monkeypatch.setattr(survey_mod.ast, "parse", counter)
        survey_repo(tmp_path, cache_path=cache)
        assert counter.calls == 4  # every file re-parsed

    def test_deleted_file_drops_from_cache(self, tmp_path):
        _write_repo(tmp_path)
        cache = tmp_path / "cache.json"
        survey_repo(tmp_path, cache_path=cache)
        (tmp_path / "nomorals" / "helper.py").unlink()
        survey_repo(tmp_path, cache_path=cache)
        cached = json.loads(cache.read_text(encoding="utf-8"))
        assert "nomorals/helper.py" not in cached["files"]
        assert "nomorals/dead.py" in cached["files"]


# ── 2. Database.write_timeout ─────────────────────────────────────────────

class _Holder:
    """Holds a write transaction open on its own thread."""

    def __init__(self, db: Database):
        self.db = db
        self.ready = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self.db.transaction():
            self.ready.set()
            self.release.wait(120)

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(30), "holder never took the transaction"
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join(30)


def _fresh_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "t.db")
    db.migrate()
    db.execute("CREATE TABLE IF NOT EXISTS wedgereal_probe "
               "(id INTEGER PRIMARY KEY, v TEXT)")
    return db


class TestWriteTimeout:
    def test_first_wait_bounded_second_fails_fast(self, tmp_path):
        db = _fresh_db(tmp_path)
        with _Holder(db):
            with db.write_timeout(0.2) as state:
                t0 = time.monotonic()
                with pytest.raises(StorageError):
                    db.execute("INSERT INTO wedgereal_probe (v) VALUES (?)",
                               ("one",))
                first = time.monotonic() - t0
                t0 = time.monotonic()
                with pytest.raises(StorageError):
                    db.execute("INSERT INTO wedgereal_probe (v) VALUES (?)",
                               ("two",))
                second = time.monotonic() - t0
            assert state.deferred == 2
            assert state.wedged is True
            assert first >= 0.15  # the first wait ran the full bound
            assert second < 0.15  # later waits fail fast, no re-proving
        # the timed-out writes never apply late
        rows = db.query("SELECT v FROM wedgereal_probe")
        assert rows == []
        # the override is restored after the block
        assert getattr(db._local, "write_timeout_s", None) is None

    def test_write_timeout_nests_and_restores(self, tmp_path):
        db = _fresh_db(tmp_path)
        with db.write_timeout(1.0):
            assert db._local.write_timeout_s == 1.0
            with db.write_timeout(0.2):
                assert db._local.write_timeout_s == 0.2
            assert db._local.write_timeout_s == 1.0
        assert getattr(db._local, "write_timeout_s", None) is None


# ── 3. watch tick with a wedged writer ────────────────────────────────────

class _FakeGateway:
    """A live chat gateway (telegram:12345 is the owner's DM)."""

    def __init__(self):
        self.sent: list[dict] = []

    def status(self):
        return {"telegram": {"running_in_session": True}}

    def send(self, platform, chat, text, **kw):
        started = time.monotonic()
        self.sent.append({"platform": platform, "text": text,
                          "started": started, "ended": started})
        self.sent[-1]["ended"] = time.monotonic()
        # the Notifier reads .ok / .message_id attributes, not a dict
        return SimpleNamespace(ok=True, message_id="1")

    def send_file(self, *args, **kwargs):
        return {"ok": True}


def _tick_context(db: Database, home: Path):
    gateway = _FakeGateway()
    settings = SimpleNamespace(
        home=str(home),
        partner=SimpleNamespace(owner_chats="telegram:12345"))
    ctx = SimpleNamespace(db=db, gateway=gateway, settings=settings,
                          workspace=None)
    return ctx, gateway


def _make_stuck_missions(db: Database, count: int = 5) -> list[str]:
    """RUNNING missions with a stale heartbeat + dead pid (the shape
    the ops scan calls stuck). The last one is already-healed. The
    plan is an empty LIST (the persisted plan shape — a list of step
    dicts, see _serialize_plan); a dict here breaks the background
    resume's _rehydrate_plan."""
    from nomorals.missions.idempotency import IdempotencyStore
    from nomorals.missions.mission import MissionStatus, MissionStore
    from nomorals.storage.kv import KVStore

    # The idempotency table is created by the store itself on first use
    # and exists on any running Devon — create it while the writer is
    # free, as production has it.
    IdempotencyStore(db)
    store = MissionStore(db)
    ids = []
    for i in range(count):
        m = store.create_new(
            f"wedgereal stuck mission {i}",
            state={"heartbeat": {"at": time.time() - 100000,
                                "pid": 999999},
                   "plan": []})
        m.status = MissionStatus.RUNNING
        store.save(m)
        ids.append(m.id)
    KVStore(db).set_raw(f"opsselfheal:stuck:{ids[-1]}",
                        str(time.time()), "json")
    return ids


class TestWatchTickWriteTimeout:
    def test_tick_completes_with_wedged_writer(self, tmp_path):
        from nomorals.agents.watcher import WatchLoop

        db = Database(tmp_path / "t.db")
        db.migrate()
        ctx, gateway = _tick_context(db, tmp_path)
        ids = _make_stuck_missions(db)
        with _Holder(db):
            t0 = time.monotonic()
            report = WatchLoop(ctx).tick(deadline_s=60.0)
            took = time.monotonic() - t0
        assert took < 10.0  # not 30s per write toward the 300s wall
        assert report["partial"] is False
        # the writer is wedged, so every self-heal save fails fast and
        # every stuck mission is reported (none is healed in the
        # background — there is no background thread to run)
        assert report["findings"] == len(ids)
        assert report["writes_deferred"] > 0
        assert len(gateway.sent) == len(ids)

    def test_tick_no_deferrals_when_writer_healthy(self, tmp_path):
        from nomorals.agents.watcher import WatchLoop

        db = Database(tmp_path / "t.db")
        db.migrate()
        ctx, gateway = _tick_context(db, tmp_path)
        _make_stuck_missions(db)
        report = WatchLoop(ctx).tick(deadline_s=60.0)
        assert report["partial"] is False
        # healthy writer: the four fresh missions self-heal in the
        # background (attempted, so not findings); only the
        # already-healed one pages
        assert report["findings"] == 1
        assert report["writes_deferred"] == 0
        # at least the one page went out synchronously inside the tick;
        # the background resumes may escalate (and send) after the tick
        # returns, so the count is a lower bound, not exact
        assert len(gateway.sent) >= 1


# ── 4. a slow job does not hold up the rest ───────────────────────────────

class _SleepingGateway(_FakeGateway):
    """Sleeps 2s on sends whose text carries the slow marker."""

    def send(self, platform, chat, text, **kw):
        if "slow-marker" in text:
            time.sleep(2.0)
        started = time.monotonic() - (2.0 if "slow-marker" in text else 0.0)
        self.sent.append({"platform": platform, "text": text,
                          "started": started,
                          "ended": time.monotonic()})
        return SimpleNamespace(ok=True, message_id="1")


class TestSchedulerIsolation:
    def test_slow_job_does_not_hold_up_fast_job(self, tmp_path):
        from nomorals.agents.scheduler import Scheduler

        db = Database(tmp_path / "t.db")
        db.migrate()
        gateway = _SleepingGateway()
        settings = SimpleNamespace(
            home=str(tmp_path),
            partner=SimpleNamespace(owner_chats="telegram:12345"))
        ctx = SimpleNamespace(db=db, gateway=gateway, settings=settings,
                              workspace=None)
        sched = Scheduler(ctx, gateway=gateway, max_concurrent=2,
                          tick_seconds=5.0)
        sched.add("wedgereal-slow", "every 60s", "message",
                  {"text": "slow-marker hello"})
        sched.add("wedgereal-fast", "every 60s", "message",
                  {"text": "fast hello"})
        past = time.time() - 10.0  # due, inside the missed-fire grace
        db.execute("UPDATE schedule_jobs SET next_run = ? "
                   "WHERE name IN ('wedgereal-slow', 'wedgereal-fast')",
                   (past,))
        t0 = time.monotonic()
        outcomes = sched.tick()
        took = time.monotonic() - t0
        assert len(outcomes) == 2
        # the two jobs overlapped: the whole tick took one slow job,
        # not the sum of both
        assert took < 3.5
        by_text = {}
        for s in gateway.sent:
            if "slow-marker hello" in s["text"]:
                by_text["slow-marker hello"] = s
            elif "fast hello" in s["text"]:
                by_text["fast hello"] = s
        assert "fast hello" in by_text and "slow-marker hello" in by_text
        # the slow send really took ~2s, and the fast job's send
        # COMPLETED while the slow one was still running
        slow = by_text["slow-marker hello"]
        fast = by_text["fast hello"]
        assert slow["ended"] - slow["started"] >= 1.9
        assert fast["ended"] < slow["ended"]
        rows = {r["name"]: r for r in db.query(
            "SELECT name, last_run FROM schedule_jobs "
            "WHERE name IN ('wedgereal-slow', 'wedgereal-fast')")}
        # both runs recorded (last_run is the execution START time, so
        # it does not order completions — the send end-times above do)
        assert rows["wedgereal-fast"]["last_run"]
        assert rows["wedgereal-slow"]["last_run"]
