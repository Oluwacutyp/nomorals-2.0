"""Tail-56 pins for nomorals/workspace/rooms.py —
2 converts / 5 exclusions / 1 flagged (576; tail-56 amendment:
602 converted on parent ruling, 576 accepted as documented
tradeoff).

10 tests: 8 site tests + 1 rehydration-mitigation test for the
flagged site 576 + 1 extra 602 test (index non-overwrite /
genuine-empty control). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

rooms.py calls no failures-ledger record function anywhere
(grep-proven: no `record_` call in the file), so NOT-recording
is pinned against the module's own observability surface: the
rooms / room_links DB tables snapshot before/after each
trigger — no row written, no row mutated beyond the site's
intended operation.

Verdict notes (full grounds in the tail-56 SURVEY.md):
- 375: read_log OSError -> collected entries ([] on a read
  failure); a missing/unreadable log is documented as not an
  error and brief() simply omits the activity section.
- 546: _slug_taken DB failure -> False; the anticipated case
  is "table may not exist yet" (the site's own comment), the
  directory check is the primary taken-test, and create()
  proceeds on disk with _persist logging any real failure.
- 576: FLAGGED (observed only) — get DB failure -> None, the
  genuine no-room value; in-module primary consumers
  (enter/_set_status/tag/link) use None as the trigger for
  the disk rehydration that is this module's source of
  truth, but tools/workspace `status` renders it verbatim
  as {"error": "no room ..."}. Parent adjudication.
- 587: CONVERTED — get_by_linked DB failure now logs a
  warning and RAISES instead of returning None: goals.py /
  projects.py _maybe_auto_room CREATE a room on None, so
  the swallow duplicated rooms on a read failure (the
  tail-53 504 shape). Both production consumers already
  guard this call (record_failure / log warning) and their
  own pin suites already drive it with a throwing double.
  Genuine missing still returns None.
- 602: CONVERTED (tail-56 amendment, parent ruling) — list
  DB failure now logs a warning and returns None (signature
  list[Room] | None); [] remains the genuine-empty value.
  Consumers guard per contract: tick() records the failure
  in its summary errors (no clean-run claim), index()
  returns None and does NOT overwrite INDEX.md, review()
  returns an unavailable string, search()/stale_rooms()
  return None, tools/workspace + cmdline room render
  unavailable, morning_briefing omits the rooms section
  (omission consumer — None treated like [] there).
- 684: _reconcile_dirty OSError -> tail stays []; the room
  is still flagged with the "recovered from unclean exit"
  blocker and a warning is logged unconditionally — the
  absence of a tail is never rendered as a clean state.
- 705: _room_links DB failure -> []; ROOM.md renders links
  as (none) for that write only, the room_links rows are
  untouched in the DB, and the next successful write
  restores them (no consumer parses links back from
  ROOM.md).
- 1169: ensure_rooms_tick_job list failure -> have=[] and
  the job is scheduled; the anticipated case is the
  scheduler table not existing yet, Scheduler.list_jobs
  itself never raises for DB failures (it records and
  returns []), and the boot caller logs any add failure.
"""

import json
import logging
import shutil
from pathlib import Path
from unittest import mock

import pytest

import nomorals.agents.scheduler as schedmod
from nomorals.storage.db import Database
from nomorals.workspace.rooms import RoomManager, ensure_rooms_tick_job


def _mgr(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.migrate()
    return RoomManager(tmp_path, db=db), db


def _snap(db):
    return {
        "rooms": db.query("SELECT * FROM rooms ORDER BY slug"),
        "links": db.query("SELECT * FROM room_links ORDER BY slug_a, slug_b"),
    }


class _ExplodingRows:
    """DB wrapper: query_rows logs every call, raises on marker."""

    def __init__(self, real, marker):
        self.real, self.marker, self.log = real, marker, []

    def query_rows(self, sql, params=()):
        self.log.append(sql)
        if self.marker in sql:
            raise RuntimeError("db boom")
        return self.real.query_rows(sql, params)

    def __getattr__(self, name):
        return getattr(self.real, name)


# ── 375: read_log OSError -> collected entries ─────────────────────────

def test_375_read_log_read_failure_returns_empty(tmp_path):
    mgr, db = _mgr(tmp_path)
    room = mgr.create("Loggy")
    with mgr.enter(room.slug) as ctx:
        ctx.log("s1", "ev1", {"a": 1})
    ctx = mgr.enter(room.slug)
    before = _snap(db)
    attempts = []
    real_read = Path.read_text

    def boom_read(self, *a, **k):
        if self.name == "activity.log":
            attempts.append(str(self))
            raise OSError("gone")
        return real_read(self, *a, **k)

    with mock.patch.object(Path, "read_text", boom_read):
        got = ctx.read_log(5)
    assert attempts, "read_text was actually attempted"
    assert got == []
    assert len(ctx.read_log(5)) >= 1  # control: healthy read returns entries
    assert _snap(db) == before  # before close(), whose checkpoint persists
    ctx.close()


# ── 546: _slug_taken DB failure -> False ───────────────────────────────

def test_546_slug_taken_db_failure_returns_false(tmp_path):
    mgr, db = _mgr(tmp_path)
    room = mgr.create("Taken One")
    # the directory check short-circuits on-disk rooms; remove the
    # dir so the DB branch is the one exercised (DB-row-only slug)
    shutil.rmtree(Path(tmp_path) / "rooms" / room.slug)
    before = _snap(db)
    wrap = _ExplodingRows(db, "FROM rooms")
    mgr.db = wrap
    assert mgr._slug_taken(room.slug) is False
    assert len(wrap.log) == 1  # the SELECT was actually attempted
    mgr.db = db
    assert mgr._slug_taken(room.slug) is True   # control: row exists
    assert mgr._slug_taken("nope-slug") is False  # genuine not-taken
    assert _snap(db) == before


# ── 576: get DB failure -> None (FLAGGED, observed) ────────────────────

def test_576_get_db_failure_returns_none_observed(tmp_path):
    mgr, db = _mgr(tmp_path)
    room = mgr.create("Gettable")
    before = _snap(db)
    wrap = _ExplodingRows(db, "FROM rooms")
    mgr.db = wrap
    assert mgr.get(room.slug) is None
    assert wrap.log, "the SELECT was actually attempted"
    mgr.db = db
    assert mgr.get(room.slug).id == room.id       # control
    assert mgr.get("nope") is None                # genuine missing
    assert _snap(db) == before


def test_576_enter_rehydrates_from_disk_when_get_fails(tmp_path):
    mgr, db = _mgr(tmp_path)
    room = mgr.create("Rehydrate Me")
    wrap = _ExplodingRows(db, "SELECT * FROM rooms")
    mgr.db = wrap
    with mgr.enter(room.slug) as ctx:   # get fails -> disk rehydration
        assert ctx.room.slug == room.slug
    mgr.db = db


# ── 587: get_by_linked DB failure -> RAISES (CONVERTED) ────────────────

def test_587_get_by_linked_db_failure_raises(tmp_path, caplog):
    mgr, db = _mgr(tmp_path)
    room = mgr.create("Linked", kind="goal", linked_id="g1")
    before = _snap(db)
    wrap = _ExplodingRows(db, "linked_id")
    mgr.db = wrap
    with caplog.at_level(logging.WARNING, logger="nomorals.workspace.rooms"):
        with pytest.raises(RuntimeError, match="db boom"):
            mgr.get_by_linked("goal", "g1")
    assert len(wrap.log) == 1  # the SELECT was actually attempted
    assert "get_by_linked failed" in caplog.text
    mgr.db = db
    assert mgr.get_by_linked("goal", "g1").slug == room.slug  # control
    assert mgr.get_by_linked("goal", "nope") is None          # genuine
    assert _snap(db) == before


# ── 602: list DB failure -> None (CONVERTED, amendment) ────────────────

def test_602_list_db_failure_returns_none_and_tick_records_error(
        tmp_path, caplog):
    mgr, db = _mgr(tmp_path)
    room = mgr.create("Listable")
    before = _snap(db)
    wrap = _ExplodingRows(db, "FROM rooms")
    mgr.db = wrap
    with caplog.at_level(logging.WARNING, logger="nomorals.workspace.rooms"):
        assert mgr.list() is None
    assert wrap.log, "the SELECT was actually attempted"
    assert "room list failed" in caplog.text
    tick = mgr.tick()
    # no clean-run claim: the failure is recorded in errors, and the
    # processed lists stay literally empty (nothing was processed)
    assert tick["advanced"] == [] and tick["skipped"] == []
    assert tick["reconciled"] == []
    assert len(tick["errors"]) == 1
    assert "unavailable" in tick["errors"][0]["error"]
    # in-module string / None-contract consumers surface unavailable
    assert "unavailable" in mgr.review()
    assert mgr.search("listable") is None
    assert mgr.stale_rooms() is None
    mgr.db = db
    assert [x.slug for x in mgr.list()] == [room.slug]      # control
    assert mgr.search("listable")                           # control: hits
    assert _snap(db) == before


def test_602_index_not_overwritten_on_failure_and_empty_store_truthful(
        tmp_path):
    mgr, db = _mgr(tmp_path)
    mgr.create("Indexed")
    dest = mgr.rooms_dir / "INDEX.md"
    sentinel = "PRE-EXISTING INDEX — do not overwrite\n"
    dest.write_text(sentinel, encoding="utf-8")
    before = _snap(db)
    wrap = _ExplodingRows(db, "FROM rooms")
    mgr.db = wrap
    assert mgr.index() is None
    assert dest.read_text(encoding="utf-8") == sentinel  # not overwritten
    assert _snap(db) == before
    mgr.db = db
    # genuine-empty store: list() == [] and index writes truthfully
    mgr2, _db2 = _mgr(tmp_path / "empty")
    assert mgr2.list() == []
    idx = mgr2.index()
    assert idx is not None and "0 rooms" in idx.read_text(encoding="utf-8")


# ── 684: _reconcile_dirty OSError -> tail [] ───────────────────────────

def test_684_reconcile_dirty_read_failure_keeps_blocker_note(tmp_path):
    mgr, db = _mgr(tmp_path)
    room = mgr.create("Dirty")
    target = mgr.get(room.slug)
    logf = Path(tmp_path) / "rooms" / room.slug / "logs" / "activity.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    logf.write_text(json.dumps({"at": "t", "step": "s", "event": "e"}) + "\n")
    before = _snap(db)
    attempts = []
    real_read = Path.read_text

    def boom_read(self, *a, **k):
        if self.name == "activity.log":
            attempts.append(str(self))
            raise OSError("gone")
        return real_read(self, *a, **k)

    with mock.patch.object(Path, "read_text", boom_read):
        mgr._reconcile_dirty(target)
    assert attempts, "read_text was actually attempted"
    assert target.state["dirty_tail"] == []
    assert any("unclean" in b for b in target.state["blockers"])
    assert "dirty_reconciled_at" in target.state
    target2 = mgr.get(room.slug)
    target2.state["blockers"] = []
    mgr._reconcile_dirty(target2)  # control: healthy read fills the tail
    assert len(target2.state["dirty_tail"]) == 1
    assert _snap(db) == before


# ── 705: _room_links DB failure -> [] ──────────────────────────────────

def test_705_room_links_db_failure_returns_empty(tmp_path):
    mgr, db = _mgr(tmp_path)
    a = mgr.create("Alpha")
    b = mgr.create("Beta")
    mgr.link(a.slug, b.slug)
    assert mgr._room_links(a.slug) == [b.slug]  # control
    before = _snap(db)
    wrap = _ExplodingRows(db, "room_links")
    mgr.db = wrap
    assert mgr._room_links(a.slug) == []
    assert len(wrap.log) == 1  # the UNION SELECT was actually attempted
    mgr.db = db
    assert _snap(db) == before  # link rows themselves untouched


# ── 1169: ensure_rooms_tick_job list failure -> schedule ───────────────

def test_1169_tick_job_list_failure_still_schedules(tmp_path):
    from types import SimpleNamespace
    mgr, db = _mgr(tmp_path)
    before = _snap(db)
    calls = {"list": 0, "add": 0}

    class FakeSched:
        def __init__(self, ctx):
            pass

        def list_jobs(self):
            calls["list"] += 1
            raise RuntimeError("no table yet")

        def add(self, name, spec, kind, payload, routine=False):
            calls["add"] += 1
            return {"id": "job-9"}

    ctx = SimpleNamespace(db=db, settings=SimpleNamespace())
    with mock.patch.object(schedmod, "Scheduler", FakeSched):
        res = ensure_rooms_tick_job(ctx)
    assert calls == {"list": 1, "add": 1}  # list was actually attempted
    assert res["scheduled"] is True and res["job_id"] == "job-9"

    class FakeSched2:
        def __init__(self, ctx):
            pass

        def list_jobs(self):
            return [{"name": "rooms tick", "id": "job-1"}]

        def add(self, *a, **k):
            calls["add"] += 1
            return {"id": "x"}

    with mock.patch.object(schedmod, "Scheduler", FakeSched2):
        res2 = ensure_rooms_tick_job(ctx)
    assert res2["already_scheduled"] is True and calls["add"] == 1  # control
    assert _snap(db) == before
