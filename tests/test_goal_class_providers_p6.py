"""WS2 P6: the health, learning, build and relationships providers.

P1–P5 registered finance, trading, careers, productivity and
interests; P6 completes the registry — every class with a
measuring organ reads measured, and ``general`` stays manual by
design.  Each provider here is pinned the P3–P5 way:

* exact math from stores seeded through the owning organ's own
  writers (``HealthTimeline.log``, ``TutorSession`` +
  ``MasteryModel``, the tracker's ``add_subgoal``, the
  ``CommitmentLedger`` event log, ``GoalSystem`` itself), never
  hand-shaped guesses at the on-disk format;
* scope vs no-scope: no declared scope and no exact match reads
  ``None`` (the class falls back to its self-reported % with the
  basis labelled); a declared scope with no flow reads a measured
  0% where the measure has a denominator, never a fabricated rate;
* exclusions and unreadable rows land in the evidence, never
  silently absorbed;
* a broken store is contained: ``None`` plus a warning naming the
  provider — never an exception in a board;
* the surface flips end to end: the class board renders the goal
  as ``measured · <source>``;
* the GoalSystem ``goal_class`` field round-trips (default
  ``general``, set/read, legacy rows from the frozen base DDL),
  and the fresh-interpreter wiring pin proves all nine providers
  register on a bare package import.

The settings home is monkeypatched to a temp dir (the P5 harness
shape), so nothing here touches the owner's real stores.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.goals import GoalSystem
from nomorals.commitments.ledger import CommitmentLedger
from nomorals.goals import GoalTracker, providers
from nomorals.goals.classes import goal_progress
from nomorals.goals.surface import (
    render_class_board,
    render_goal_progress,
    run_sync,
)
from nomorals.health.timeline import HealthTimeline
from nomorals.learn.tutor import TutorSession
from nomorals.storage import Database

NOW = 1_800_000_000.0          # 2027-01-15T08:00:00Z — fixed "today"
SINCE = NOW - 7 * 86400.0      # goal filed exactly 7 days earlier


@pytest.fixture()
def env(monkeypatch):
    """A tracker over a throwaway db + a throwaway settings home."""
    tmp = tempfile.TemporaryDirectory(dir=os.environ.get("NM_TEST_TMPDIR"))
    home = Path(tmp.name)
    settings = SimpleNamespace(
        home_path=home, briefing=SimpleNamespace(topics=[]))
    monkeypatch.setattr(providers, "get_settings", lambda: settings)
    tracker = GoalTracker(Database(":memory:"))
    yield SimpleNamespace(home=home, tracker=tracker)
    tmp.cleanup()


def _goal(goal_id: str = "g1", title: str = "Goal",
          goal_class: str = "general", tags: tuple[str, ...] = (),
          created_at: float = SINCE, progress: float = 40.0,
          subgoals: list | None = None):
    return SimpleNamespace(
        goal_id=goal_id, title=title, goal_class=goal_class,
        tags=list(tags), created_at=created_at, progress=progress,
        subgoals=subgoals or [])


# ── health: check-in adherence from the timeline store ────────────────


def _seed_health(env, events: list[tuple[str, float]]) -> Path:
    path = env.home / "health" / "timeline.db"
    timeline = HealthTimeline(db_path=path)
    for event_type, ts in events:
        timeline.log(event_type, f"{event_type} entry", ts=ts)
    return path


def test_health_adherence_math_from_real_timeline(env):
    # Window: [SINCE, NOW] = 8 local days (SINCE .. NOW inclusive).
    # Events on 4 distinct days: filing day, +1d, +2d (twice), +4d.
    _seed_health(env, [
        ("note", SINCE + 3600),
        ("sleep", SINCE + 86400 + 3600),
        ("mood", SINCE + 2 * 86400 + 3600),
        ("symptom", SINCE + 2 * 86400 + 7200),
        ("measurement", SINCE + 4 * 86400 + 3600),
        ("note", SINCE - 86400),          # before the window
        ("note", NOW + 86400),            # after the reading
    ])
    goal = _goal(goal_class="health")
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.basis == "measured"
    assert prog.source == "health timeline"
    assert prog.percent == pytest.approx(4 / 8 * 100.0)
    ev = prog.evidence
    assert ev["measure"] == "checkin_adherence"
    assert ev["active_days"] == 4
    assert ev["days_in_window"] == 8
    assert ev["events_in_window"] == 5
    assert ev["events_by_type"] == {
        "note": 1, "sleep": 1, "mood": 1, "symptom": 1,
        "measurement": 1}
    assert ev["events_before_window"] == 1
    assert ev["events_after_reading"] == 1
    assert ev["last_event_at"] == SINCE + 4 * 86400 + 3600
    assert "4 of 8 day(s)" in prog.headline


def test_health_no_events_reads_measured_zero(env):
    _seed_health(env, [])  # the organ's store exists, nothing logged
    prog = goal_progress(env.tracker.db, _goal(goal_class="health"),
                         now=NOW)
    assert prog.basis == "measured"
    assert prog.percent == 0.0
    assert "nothing logged" in prog.headline
    assert prog.evidence["active_days"] == 0
    assert prog.evidence["events_in_window"] == 0


def test_health_store_absent_reads_measured_zero(env):
    # No health/ directory at all: the organ has never run — still
    # a measured 0, not a failure (and no store is created).
    prog = goal_progress(env.tracker.db, _goal(goal_class="health"),
                         now=NOW)
    assert prog.basis == "measured"
    assert prog.percent == 0.0
    assert not (env.home / "health").exists()


def test_health_events_before_window_do_not_count(env):
    _seed_health(env, [("note", SINCE - 3600), ("note", SINCE - 86400)])
    prog = goal_progress(env.tracker.db, _goal(goal_class="health"),
                         now=NOW)
    assert prog.percent == 0.0
    assert prog.evidence["events_before_window"] == 2
    assert prog.evidence["events_in_window"] == 0


def test_health_unreadable_ts_row_counted(env):
    path = _seed_health(env, [("note", SINCE + 3600)])
    con = sqlite3.connect(str(path))
    con.execute(
        "INSERT INTO health_events (event_type, text, ts) "
        "VALUES ('note', 'broken clock', 'not-a-time')")
    con.commit()
    con.close()
    prog = goal_progress(env.tracker.db, _goal(goal_class="health"),
                         now=NOW)
    assert prog.percent == pytest.approx(1 / 8 * 100.0)
    assert prog.evidence["unreadable_event_rows"] == 1
    assert prog.evidence["events_in_window"] == 1


def test_health_corrupt_store_contained(env, caplog):
    broken = env.home / "health"
    broken.mkdir(parents=True)
    (broken / "timeline.db").write_bytes(b"definitely not sqlite")
    with caplog.at_level("WARNING", logger="nomorals.goals.providers"):
        prog = providers.health_progress(
            env.tracker.db, _goal(goal_class="health"), now=NOW)
    assert prog is None
    assert "health progress provider failed" in caplog.text


def test_health_columns_mirror_the_real_timeline_schema(env):
    # The provider reads event_type/ts by name; pin those columns
    # against a store the organ itself created.
    path = _seed_health(env, [("note", SINCE)])
    con = sqlite3.connect(str(path))
    cols = {row[1] for row in con.execute("PRAGMA table_info(health_events)")}
    con.close()
    assert {"event_type", "ts", "text"} <= cols


# ── learning: BKT mastery from the tutor session files ────────────────


def _seed_session(env, topic: str, skills_bkt: dict[str, float],
                  *, started_at: float, turns: int = 3,
                  named_only: tuple[str, ...] = ()) -> Path:
    """A session persisted through the organ's own save() path.

    Skills are the session's sub-skills; posteriors are planted in
    the model's BKT table (the values ``bkt_update`` would have
    converged to), and ``named_only`` skills are carried in the
    skills table with no observation — the prior case.
    """
    skills = list(skills_bkt) + list(named_only)
    session = TutorSession(topic=topic, sub_skills=skills or ["general"])
    session._started_at = started_at
    session.turns = turns
    session.mastery._bkt.update(skills_bkt)
    path = env.home / "learn" / "sessions" / f"{session.id}.json"
    session.save(path)
    return path


def test_learning_mastery_math_from_real_sessions(env):
    _seed_session(env, "Yoruba greetings", {"kaabo": 0.8, "bawo": 0.4},
                  started_at=SINCE + 100, turns=5)
    _seed_session(env, "French verbs", {"aller": 0.9},
                  started_at=SINCE + 200)
    goal = _goal(goal_class="learning", title="Yoruba greetings")
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.basis == "measured"
    assert prog.source == "tutor mastery"
    assert prog.percent == pytest.approx((0.8 + 0.4) / 2 * 100.0)
    ev = prog.evidence
    assert ev["measure"] == "mastery"
    assert ev["link"] == "title"
    assert ev["topic"] == "Yoruba greetings"
    assert ev["sessions"] == 1
    assert ev["skills"] == {"bawo": 0.4, "kaabo": 0.8}
    assert ev["total_turns"] == 5


def test_learning_latest_session_wins_per_skill(env):
    _seed_session(env, "Piano scales", {"c-major": 0.3},
                  started_at=SINCE + 100)
    _seed_session(env, "Piano scales", {"c-major": 0.9, "g-major": 0.5},
                  started_at=SINCE + 5000)
    goal = _goal(goal_class="learning",
                 tags=["learn:Piano scales"])
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.percent == pytest.approx((0.9 + 0.5) / 2 * 100.0)
    assert prog.evidence["link"] == "learn-tag"
    assert prog.evidence["sessions"] == 2
    assert prog.evidence["skills"] == {"c-major": 0.9, "g-major": 0.5}


def test_learning_unobserved_skill_reads_the_bkt_prior(env):
    _seed_session(env, "Chess tactics", {"pins": 0.6},
                  started_at=SINCE + 100, named_only=("forks",))
    goal = _goal(goal_class="learning", tags=["learn:Chess tactics"])
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.percent == pytest.approx((0.6 + 0.10) / 2 * 100.0)
    assert prog.evidence["skills"]["forks"] == pytest.approx(0.10)


def test_learning_bkt_prior_mirror_matches_the_organ():
    from nomorals.learn.tutor import BKT_DEFAULTS
    assert BKT_DEFAULTS["p_init"] == providers._BKT_P_INIT


def test_learning_tag_with_no_sessions_reads_measured_zero(env):
    goal = _goal(goal_class="learning", tags=["learn:Swahili basics"])
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.basis == "measured"
    assert prog.percent == 0.0
    assert "no tutor sessions" in prog.headline
    assert prog.evidence["sessions"] == 0


def test_learning_no_scope_reads_none(env):
    _seed_session(env, "French verbs", {"aller": 0.9},
                  started_at=SINCE + 200)
    goal = _goal(goal_class="learning", title="Something else entirely")
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    # No scope: the manual % stands, labelled self-reported.
    assert prog.basis == "self-reported"
    assert prog.source == "manual"
    assert prog.percent == 40.0


def test_learning_ambiguous_title_match_reads_none(env):
    _seed_session(env, "Yoruba Greetings", {"kaabo": 0.5},
                  started_at=SINCE + 100)
    _seed_session(env, "yoruba greetings", {"kaabo": 0.7},
                  started_at=SINCE + 200)
    # Two distinct stored topics normalise to the goal's title:
    # the P3 rule — never guess between them.
    goal = _goal(goal_class="learning", title="YORUBA GREETINGS")
    assert providers.learning_progress(
        env.tracker.db, goal, now=NOW) is None


def test_learning_unreadable_session_file_counted(env):
    _seed_session(env, "Piano scales", {"c-major": 0.5},
                  started_at=SINCE + 100)
    sessions = env.home / "learn" / "sessions"
    (sessions / "junk.json").write_text("{not json", encoding="utf-8")
    (sessions / "bare.json").write_text("[1, 2]", encoding="utf-8")
    goal = _goal(goal_class="learning", tags=["learn:Piano scales"])
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.percent == pytest.approx(50.0)
    assert prog.evidence["unreadable_session_files"] == 2


def test_learning_sessions_predating_goal_disclosed(env):
    _seed_session(env, "Piano scales", {"c-major": 0.5},
                  started_at=SINCE - 86400)
    goal = _goal(goal_class="learning", tags=["learn:Piano scales"])
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    # Mastery is state: pre-goal knowledge still counts, disclosed.
    assert prog.percent == pytest.approx(50.0)
    assert prog.evidence["sessions_predating_goal"] == 1


def test_learning_store_failure_contained(env, caplog, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("sessions dir unreadable")
    monkeypatch.setattr(providers, "_learn_session_records", boom)
    with caplog.at_level("WARNING", logger="nomorals.goals.providers"):
        prog = providers.learning_progress(
            env.tracker.db,
            _goal(goal_class="learning", tags=["learn:X"]), now=NOW)
    assert prog is None
    assert "learning progress provider failed" in caplog.text


# ── build: execution steps ────────────────────────────────────────────


def _tracker_goal_with_subgoals(env, *, total: int, done: int,
                                goal_class: str = "build") -> str:
    goal = run_sync(env.tracker.create(
        "user", "Build the thing", goal_class=goal_class))
    for i in range(total):
        run_sync(env.tracker.add_subgoal(goal.goal_id, f"step {i + 1}"))
    subs = run_sync(env.tracker.get(goal.goal_id)).subgoals
    for sub in subs[:done]:
        run_sync(env.tracker.complete_subgoal(sub.subgoal_id))
    return goal.goal_id


def test_build_subgoal_steps_math(env):
    gid = _tracker_goal_with_subgoals(env, total=4, done=3)
    goal = run_sync(env.tracker.get(gid))
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.basis == "measured"
    assert prog.source == "goal subgoals"
    assert prog.percent == pytest.approx(75.0)
    assert prog.evidence["steps_done"] == 3
    assert prog.evidence["steps_total"] == 4
    assert "3 of 4 step(s)" in prog.headline


def test_build_no_steps_reads_none(env):
    goal = run_sync(env.tracker.create(
        "user", "No steps yet", goal_class="build"))
    prog = goal_progress(
        env.tracker.db, run_sync(env.tracker.get(goal.goal_id)), now=NOW)
    assert prog.basis == "self-reported"  # nothing recorded to measure


def _migrated_tracker() -> tuple[Database, GoalTracker]:
    """A tracker + GoalSystem over ONE migrated database — the
    production shape (both organs share ``context.db``)."""
    db = Database(":memory:")
    db.migrate()
    return db, GoalTracker(db)


def test_build_linked_agent_goal_steps_math(env):
    db, _tracker = _migrated_tracker()
    system = GoalSystem(SimpleNamespace(db=db))
    agent_goal = system.create("Agent build", plan=["one", "two", "three"])
    steps = system.get(agent_goal.id).steps
    system._set_step(steps[0].id, status="done")
    system._set_step(steps[1].id, status="done")
    # third step stays pending
    goal = _goal(goal_class="build",
                 tags=[f"agent-goal:{agent_goal.id}"])
    prog = providers.build_progress(db, goal, now=NOW)
    assert prog.basis == "measured"
    assert prog.source == "agent goal steps"
    assert prog.percent == pytest.approx(2 / 3 * 100.0)
    assert prog.evidence["link"] == "agent-goal"
    assert prog.evidence["steps_done"] == 2
    assert prog.evidence["steps_total"] == 3


def test_build_skipped_steps_leave_the_denominator(env):
    db, _tracker = _migrated_tracker()
    system = GoalSystem(SimpleNamespace(db=db))
    agent_goal = system.create("Agent build", plan=["one", "two", "three"])
    steps = system.get(agent_goal.id).steps
    system._set_step(steps[0].id, status="done")
    system._set_step(steps[1].id, status="skipped")              # dropped
    goal = _goal(goal_class="build",
                 tags=[f"agent-goal:{agent_goal.id}"])
    prog = providers.build_progress(db, goal, now=NOW)
    assert prog.percent == pytest.approx(1 / 2 * 100.0)
    assert prog.evidence["steps_skipped"] == 1
    assert prog.evidence["steps_by_status"]["skipped"] == 1


def test_build_stale_agent_tag_reads_none(env):
    db, tracker = _migrated_tracker()
    goal = run_sync(tracker.create(
        "user", "Linked build", goal_class="build",
        tags=["agent-goal:does-not-exist"]))
    run_sync(tracker.add_subgoal(goal.goal_id, "own step"))
    prog = providers.build_progress(
        db, run_sync(tracker.get(goal.goal_id)), now=NOW)
    # A deliberate-but-stale link measures nothing — it does NOT
    # fall through to the goal's own subgoals (the P3 rule).
    assert prog is None


def test_build_store_failure_contained(env, caplog):
    class ExplodingDb:
        def query(self, *args, **kwargs):
            raise RuntimeError("db gone")

        def query_one(self, *args, **kwargs):
            raise RuntimeError("db gone")
    with caplog.at_level("WARNING", logger="nomorals.goals.providers"):
        prog = providers.build_progress(
            ExplodingDb(), _goal(goal_class="build"), now=NOW)
    assert prog is None
    assert "build progress provider failed" in caplog.text


# ── GoalSystem goal_class field ───────────────────────────────────────


def _migrated_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def test_goal_system_class_defaults_to_general():
    system = GoalSystem(SimpleNamespace(db=_migrated_db()))
    goal = system.create("Unclassed", plan=["step"])
    assert goal.goal_class == "general"
    assert system.get(goal.id).goal_class == "general"
    assert goal.to_dict()["goal_class"] == "general"


def test_goal_system_class_round_trip():
    system = GoalSystem(SimpleNamespace(db=_migrated_db()))
    goal = system.create("Classed", plan=["step"], goal_class="build")
    assert goal.goal_class == "build"
    assert system.get(goal.id).goal_class == "build"
    refilled = system.set_goal_class(goal.id, "finance")
    assert refilled.goal_class == "finance"
    assert system.get(goal.id).goal_class == "finance"
    cleared = system.set_goal_class(goal.id, None)
    assert cleared.goal_class == "general"
    assert system.set_goal_class("missing", "build") is None


def test_goal_system_class_validated():
    system = GoalSystem(SimpleNamespace(db=_migrated_db()))
    with pytest.raises(ValueError):
        system.create("Bad class", plan=["step"], goal_class="nope")
    goal = system.create("Good class", plan=["step"])
    with pytest.raises(ValueError):
        system.set_goal_class(goal.id, "nope")
    assert system.get(goal.id).goal_class == "general"


def test_goal_system_legacy_rows_read_general():
    db = _migrated_db()
    columns = {row["name"] for row in
               db.query("PRAGMA table_info(agent_goals)")}
    # The frozen migration DDL carries no goal_class column; the
    # field is a runtime defensive add (the deadline pattern).
    assert "goal_class" not in columns
    db.execute(
        "INSERT INTO agent_goals (id, title, status, progress, "
        "priority, heals, created_at, updated_at) "
        "VALUES ('legacy1', 'Old goal', 'active', 0.5, 0, 0, 1.0, 1.0)")
    system = GoalSystem(SimpleNamespace(db=db))
    legacy = system.get("legacy1")
    assert legacy.goal_class == "general"
    assert system.set_goal_class("legacy1", "learning").goal_class \
        == "learning"
    fresh = system.create("New goal", plan=["s"], goal_class="health")
    assert fresh.goal_class == "health"


def test_goal_system_tool_carries_the_class():
    from nomorals.agents.goals import register

    class _Registry:
        """The registry surface register() decorates against."""

        def __init__(self, context):
            self.context = context
            self.tools: dict = {}

        def register(self, name, description="", capability="",
                     parameters=None):
            def deco(fn):
                self.tools[name] = fn
                return fn
            return deco

    registry = _Registry(SimpleNamespace(db=_migrated_db()))
    register(registry)
    made = registry.tools["goal"]("create", title="Tool build",
                                  goal_class="build")
    assert made["ok"] is True
    assert made["goal"]["goal_class"] == "build"
    bad = registry.tools["goal"]("create", title="Tool bad",
                                 goal_class="nope")
    assert bad["ok"] is False
    assert "unknown goal class" in bad["error"]


# ── relationships: kept promises to one person ────────────────────────


def _seed_commitments(env, specs: list[dict]) -> Path:
    """Seed the ledger through its own append-only API: ``add``
    for the promise, then the lifecycle verb for its verdict —
    the persisted event log is exactly what production writes."""
    path = env.home / "data" / "commitments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = CommitmentLedger(str(path))
    for spec in specs:
        commitment = ledger.add(
            promisor=spec.get("promisor", "owner"),
            promisee=spec["promisee"], what=f"promise {spec['id']}",
            deadline=spec["deadline"], ts=SINCE)
        status = spec["status"]
        if status == "kept":
            ledger.mark_kept(commitment.id, ts=NOW - 60)
        elif status == "broken":
            ledger.mark_broken(commitment.id, ts=NOW - 60)
        elif status == "cancelled":
            ledger.cancel(commitment.id, ts=NOW - 60)
    return path


def _entry(cid: str, promisee: str, status: str,
           deadline: float | None, *, promisor: str = "owner") -> dict:
    return {"id": cid, "promisee": promisee, "status": status,
            "deadline": deadline, "promisor": promisor}


def test_relationships_math_for_the_scoped_person(env):
    _seed_commitments(env, [
        _entry("c1", "Mum", "kept", SINCE + 86400),
        _entry("c2", "Mum", "kept", SINCE + 2 * 86400),
        _entry("c3", "Mum", "broken", SINCE + 3 * 86400),
        _entry("c4", "Mum", "open", SINCE + 4 * 86400),     # overdue
        _entry("c5", "Dad", "kept", SINCE + 86400),         # other person
        _entry("c6", "Dad", "broken", SINCE + 86400),       # other person
        _entry("c7", "Mum", "cancelled", SINCE + 86400),    # not a miss
        _entry("c8", "Mum", "open", NOW + 86400),           # not yet due
        _entry("c9", "Mum", "kept", SINCE - 86400),         # before window
        _entry("c10", "Mum", "kept", SINCE + 86400,
               promisor="someone-else"),                    # not owner's
    ])
    goal = _goal(goal_class="relationships", tags=["person:Mum"])
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.basis == "measured"
    assert prog.source == "commitment ledger"
    assert prog.percent == pytest.approx(2 / 4 * 100.0)
    ev = prog.evidence
    assert ev["person"] == "Mum"
    assert ev["link"] == "person-tag"
    assert ev["kept"] == 2 and ev["broken"] == 1
    assert ev["overdue_open"] == 1 and ev["due"] == 4
    assert ev["excluded"] == {
        "cancelled": 1, "not_yet_due": 1, "due_before_window": 1,
        "other_promisor": 1, "other_promisee": 2}
    assert "kept 2 of 4 promise(s) to Mum" in prog.headline


def test_relationships_title_scope_matches_promisee(env):
    _seed_commitments(env, [
        _entry("c1", "Aunt Rose", "kept", SINCE + 86400),
        _entry("c2", "Aunt Rose", "broken", SINCE + 2 * 86400),
    ])
    goal = _goal(goal_class="relationships", title="aunt rose")
    prog = goal_progress(env.tracker.db, goal, now=NOW)
    assert prog.percent == pytest.approx(50.0)
    assert prog.evidence["link"] == "title"
    assert prog.evidence["person"] == "Aunt Rose"


def test_relationships_no_scope_reads_none(env):
    _seed_commitments(env, [
        _entry("c1", "Mum", "kept", SINCE + 86400)])
    goal = _goal(goal_class="relationships", title="Be nicer")
    assert providers.relationships_progress(
        env.tracker.db, goal, now=NOW) is None


def test_relationships_nothing_due_reads_none(env):
    _seed_commitments(env, [
        _entry("c1", "Mum", "open", NOW + 86400),       # not yet due
        _entry("c2", "Mum", "cancelled", SINCE + 100),  # not a miss
        _entry("c3", "Dad", "kept", SINCE + 100),       # other person
    ])
    goal = _goal(goal_class="relationships", tags=["person:Mum"])
    assert providers.relationships_progress(
        env.tracker.db, goal, now=NOW) is None


def test_relationships_corrupt_ledger_contained(env, caplog):
    path = env.home / "data"
    path.mkdir(parents=True)
    (path / "commitments.json").write_text("{broken", encoding="utf-8")
    with caplog.at_level("WARNING", logger="nomorals.goals.providers"):
        prog = providers.relationships_progress(
            env.tracker.db,
            _goal(goal_class="relationships", tags=["person:Mum"]),
            now=NOW)
    assert prog is None
    assert "relationships progress provider failed" in caplog.text


# ── surface flips: the board renders all four as measured ─────────────


def _filed_at(env, goal_id: str, when: float = SINCE) -> None:
    """Pin a tracker goal's filing time (the window measures read)."""
    env.tracker.db.execute(
        "UPDATE goals SET created_at = ? WHERE goal_id = ?",
        (when, goal_id))


def test_surface_flips_for_all_four_classes(env):
    # health — one event today in an 8-day window
    _seed_health(env, [("note", SINCE + 3600)])
    health = run_sync(env.tracker.create(
        "user", "Keep the health log", goal_class="health"))
    _filed_at(env, health.goal_id)
    # learning — one session on the goal's own topic
    _seed_session(env, "Piano scales", {"c-major": 0.5},
                  started_at=SINCE + 100)
    learning = run_sync(env.tracker.create(
        "user", "Piano scales", goal_class="learning"))
    # build — 1 of 2 subgoals done
    build_id = _tracker_goal_with_subgoals(env, total=2, done=1)
    # relationships — 1 kept of 2 due to Mum
    _seed_commitments(env, [
        _entry("c1", "Mum", "kept", SINCE + 86400),
        _entry("c2", "Mum", "broken", SINCE + 2 * 86400),
    ])
    rel = run_sync(env.tracker.create(
        "user", "Show up for Mum", goal_class="relationships",
        tags=["person:Mum"]))
    _filed_at(env, rel.goal_id)
    board = render_class_board(env.tracker, "user", now=NOW)
    assert "12% (measured · health timeline)" in board
    assert "50% (measured · tutor mastery)" in board
    assert "50% (measured · goal subgoals)" in board
    assert "50% (measured · commitment ledger)" in board
    line = render_goal_progress(env.tracker, "user", rel.goal_id, now=NOW)
    assert "measured · commitment ledger" in line
    assert "kept 1 of 2 promise(s) to Mum" in line
    health_line = render_goal_progress(
        env.tracker, "user", health.goal_id, now=NOW)
    assert "measured · health timeline" in health_line
    learning_line = render_goal_progress(
        env.tracker, "user", learning.goal_id, now=NOW)
    assert "measured · tutor mastery" in learning_line
    build_line = render_goal_progress(
        env.tracker, "user", build_id, now=NOW)
    assert "measured · goal subgoals" in build_line


# ── wiring: a bare import registers all nine providers ────────────────


def test_fresh_interpreter_registers_nine_providers():
    code = (
        "import json\n"
        "import nomorals.goals\n"
        "from nomorals.goals.classes import list_classes, "
        "registered_provider\n"
        "print(json.dumps({c.id: registered_provider(c.id) is not None "
        "for c in list_classes()}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ,
             "NM_TEST_TMPDIR": os.environ.get("NM_TEST_TMPDIR", "")},
        timeout=120)
    assert proc.returncode == 0, proc.stderr
    states = json.loads(proc.stdout.strip().splitlines()[-1])
    assert states == {
        "careers": True, "productivity": True, "finance": True,
        "interests": True, "health": True, "learning": True,
        "build": True, "relationships": True, "trading": True,
        "general": False,
    }
