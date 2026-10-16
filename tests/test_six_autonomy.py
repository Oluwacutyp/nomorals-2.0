"""Tests for the six new autonomy modules: unified, budget, why,
rhythm, usage, preemption."""

import datetime as _dt
import json

import pytest

from nomorals.autonomy import unified as _unified
from nomorals.autonomy import budget as _budget
from nomorals.autonomy import why as _why
from nomorals.autonomy import rhythm as _rhythm
from nomorals.autonomy import usage as _usage
from nomorals.autonomy import preemption as _preemption


# ── UnifiedScheduler ─────────────────────────────────────────────────

class TestUnifiedScheduler:
    def test_mission_survives_kill9(self, tmp_path):
        """Schedule, drop the scheduler (simulated kill -9), reload:
        the mission is still pending with its plan intact."""
        queue = tmp_path / "queue.json"
        sched = _unified.UnifiedScheduler(queue)
        mid = sched.schedule_mission({
            "goal": "prepare morning briefing",
            "plan": [{"name": "fetch", "handler": "fetch"},
                     {"name": "summarize", "handler": "summarize"}],
        })
        del sched  # kill -9: no shutdown, no flush hook, just gone

        sched2 = _unified.UnifiedScheduler(queue)
        pending = sched2.pending_missions()
        assert [m["id"] for m in pending] == [mid]
        assert [s["name"] for s in pending[0]["plan"]] == ["fetch", "summarize"]

        ran = []
        sched2.register_handler("fetch", lambda m, s: ran.append("fetch"))
        sched2.register_handler("summarize", lambda m, s: ran.append("summarize"))
        results = sched2.run_due()
        assert ran == ["fetch", "summarize"]
        assert results[0]["status"] == "done"
        assert sched2.get_mission(mid)["status"] == "done"

    def test_mission_without_handler_skips_honestly(self, tmp_path):
        sched = _unified.UnifiedScheduler(tmp_path / "q.json")
        mid = sched.schedule_mission(
            {"goal": "do thing", "plan": [{"name": "x", "handler": "missing"}]})
        results = sched.run_due()
        assert results[0]["status"] == "done"
        assert results[0]["steps"][0]["status"] == "skipped"
        assert "no handler" in results[0]["steps"][0]["reason"]
        assert sched.get_mission(mid)["status"] == "done"

    def test_organ_tick_runs_on_interval(self, tmp_path):
        sched = _unified.UnifiedScheduler(tmp_path / "q.json")
        fired = []
        sched.register_organ_tick("heartbeat", lambda: fired.append(1),
                                  interval=60, start_now=1000.0)
        assert sched.run_due(now=1059.0) == []
        results = sched.run_due(now=1060.0)
        assert fired == [1]
        assert results[0] == {"kind": "tick", "name": "heartbeat",
                              "status": "ok", "outcome": ""}
        assert sched.run_due(now=1061.0) == []

    def test_tick_failure_does_not_kill_run_due(self, tmp_path):
        sched = _unified.UnifiedScheduler(tmp_path / "q.json")

        def boom():
            raise RuntimeError("tick exploded")

        ok = []
        sched.register_organ_tick("bad", boom, interval=10, start_now=0.0)
        sched.register_organ_tick("good", lambda: ok.append(1),
                                  interval=10, start_now=0.0)
        results = sched.run_due(now=100.0)
        assert {r["name"]: r["status"] for r in results} == {
            "bad": "failed", "good": "ok"}
        assert ok == [1]

    def test_rhythm_gate_defers_in_quiet_hours(self, tmp_path):
        rhythm = _rhythm.OwnerRhythm(quiet_start=22, quiet_end=7)
        sched = _unified.UnifiedScheduler(tmp_path / "q.json", rhythm=rhythm)
        sched.register_organ_tick("ambient-job", lambda: None, interval=10,
                                  priority="ambient", start_now=0.0)
        at_2am = _dt.datetime(2026, 10, 16, 2, 0).timestamp()
        results = sched.run_due(now=at_2am)
        assert results[0]["status"] == "deferred"

    def test_budget_gate_defers_over_ceiling(self, tmp_path):
        gov = _budget.BudgetGovernor(daily_token_ceiling=10)
        gov.track_spend("research", tokens=10)
        sched = _unified.UnifiedScheduler(tmp_path / "q.json", budget=gov)
        mid = sched.schedule_mission({"goal": "research more", "organ": "research"})
        results = sched.run_due()
        assert results[0]["status"] == "deferred"
        assert "ceiling" in results[0]["reason"]
        assert sched.get_mission(mid)["status"] == "pending"

    def test_why_recorded_on_firing(self, tmp_path):
        why = _why.WhyStore()
        sched = _unified.UnifiedScheduler(tmp_path / "q.json", why_store=why)
        sched.schedule_mission({"goal": "explain me"})
        sched.run_due()
        last = why.why_last()
        assert last["found"] is True
        assert "mission:" in last["what_fired"]

    def test_queue_file_is_valid_json_after_kill(self, tmp_path):
        queue = tmp_path / "q.json"
        sched = _unified.UnifiedScheduler(queue)
        sched.schedule_mission({"goal": "a"})
        sched.schedule_mission({"goal": "b"})
        raw = json.loads(queue.read_text(encoding="utf-8"))
        assert raw["version"] == 1
        assert [m["goal"] for m in raw["missions"]] == ["a", "b"]


# ── BudgetGovernor ───────────────────────────────────────────────────

class TestBudgetGovernor:
    def test_check_allows_within_budget(self):
        gov = _budget.BudgetGovernor(
            daily_token_ceiling=1000,
            per_organ_budgets={"research": {"tokens": 100}})
        gov.track_spend("research", tokens=10)
        verdict = gov.check("research")
        assert verdict["allowed"] is True
        assert verdict["status"] == "allowed"

    def test_daily_ceiling_defers_everyone(self):
        gov = _budget.BudgetGovernor(daily_token_ceiling=50)
        gov.track_spend("research", tokens=50)
        assert gov.check("research")["allowed"] is False
        assert gov.check("other-organ")["allowed"] is False

    def test_per_organ_budget_preempts_over_ceiling(self):
        """Over-ceiling organ work is deferred and preemption is logged."""
        gov = _budget.BudgetGovernor(
            per_organ_budgets={"background": {"tokens": 5}})
        gov.track_spend("background", tokens=6)
        verdict = gov.check("background")
        assert verdict["status"] == "deferred"
        assert "background" in verdict["reason"]
        record = gov.preempt("background-job",
                             preempted_by="scheduled briefing",
                             reason="background over token budget")
        assert record["victim"] == "background-job"
        assert gov.preemptions_for("background-job")[0]["id"] == record["id"]
        # The healthy organ is untouched.
        assert gov.check("research")["allowed"] is True

    def test_track_spend_time_alias(self):
        gov = _budget.BudgetGovernor()
        snap = gov.track_spend("x", tokens=3, cost=0.5, time=12.0)
        assert snap["seconds"] == 12.0
        assert snap["tokens"] == 3

    def test_unknown_organ_gets_daily_ceiling_only(self):
        gov = _budget.BudgetGovernor(daily_cost_ceiling=1.0)
        assert gov.check("never-seen")["allowed"] is True
        gov.track_spend("never-seen", cost=1.0)
        assert gov.check("never-seen")["allowed"] is False


# ── WhyStore / nm why ────────────────────────────────────────────────

class TestWhy:
    def test_why_always_answers_empty_store(self):
        why = _why.WhyStore()
        last = why.why_last()
        assert last["found"] is False
        assert "no autonomous firing" in last["reason"]
        text = _why.why_command(why)
        assert "no record" in text

    def test_why_always_answers_unknown_id(self):
        why = _why.WhyStore()
        why.record("tick:a", evidence={"x": 1}, did="ran", happened="ok")
        missing = why.why_for("fire-does-not-exist")
        assert missing["found"] is False
        assert missing["id"] == "fire-does-not-exist"
        text = _why.why_command(why, "fire-does-not-exist")
        assert "no record" in text
        assert "fire-does-not-exist" in text

    def test_why_records_and_explains(self):
        why = _why.WhyStore()
        fid = why.record("mission:briefing",
                         evidence={"goal": "morning briefing"},
                         model="llama-3.3-70b",
                         did="built the morning briefing",
                         happened="delivered 3 items")
        got = why.why_for(fid)
        assert got["found"] is True
        assert got["model"] == "llama-3.3-70b"
        assert got["did"] == "built the morning briefing"
        assert got["happened"] == "delivered 3 items"
        text = _why.why_command(why)
        assert "morning briefing" in text
        assert "llama-3.3-70b" in text

    def test_why_persists_across_reload(self, tmp_path):
        path = tmp_path / "why.jsonl"
        why = _why.WhyStore(path)
        fid = why.record("tick:x", did="ran", happened="ok")
        why2 = _why.WhyStore(path)
        assert why2.why_for(fid)["found"] is True
        assert len(why2) == 1


# ── OwnerRhythm ──────────────────────────────────────────────────────

class TestOwnerRhythm:
    def _at(self, hour):
        return _dt.datetime(2026, 10, 16, hour, 0)

    def test_quiet_hours_suppress_firing(self):
        rhythm = _rhythm.OwnerRhythm(quiet_start=22, quiet_end=7)
        assert rhythm.is_quiet(self._at(2)) is True
        assert rhythm.is_quiet(self._at(23)) is True
        assert rhythm.is_quiet(self._at(12)) is False
        assert rhythm.should_fire(self._at(2), priority="ambient") is False
        assert rhythm.should_fire(self._at(2), priority="background") is False
        assert rhythm.should_fire(self._at(2), kind="serendipity") is False
        # Daytime fires.
        assert rhythm.should_fire(self._at(12), priority="ambient") is True

    def test_quiet_hours_never_block_safety_or_owner(self):
        rhythm = _rhythm.OwnerRhythm(quiet_start=22, quiet_end=7)
        assert rhythm.should_fire(self._at(3), priority="safety") is True
        assert rhythm.should_fire(self._at(3), priority="owner-direct") is True
        assert rhythm.should_fire(self._at(3), priority="scheduled") is True

    def test_serendipity_held_till_morning(self):
        rhythm = _rhythm.OwnerRhythm(quiet_start=22, quiet_end=7)
        rhythm.hold_serendipity({"title": "interesting find"})
        assert rhythm.release_at_morning(self._at(3)) == []
        assert len(rhythm.held_serendipity()) == 1
        released = rhythm.release_at_morning(self._at(8))
        assert len(released) == 1
        assert released[0]["item"]["title"] == "interesting find"
        assert rhythm.held_serendipity() == []

    def test_busy_and_low_intensity(self):
        rhythm = _rhythm.OwnerRhythm(busy=True)
        assert rhythm.should_fire(self._at(12), priority="ambient") is False
        assert rhythm.should_fire(self._at(12), priority="scheduled") is True
        rhythm.set_busy(False)
        rhythm.set_intensity("low")
        assert rhythm.should_fire(self._at(12), priority="ambient") is False
        assert rhythm.should_fire(self._at(12), priority="scheduled") is True
        rhythm.set_intensity("high")
        rhythm.set_busy(True)
        assert rhythm.should_fire(self._at(12), priority="ambient") is True
        with pytest.raises(ValueError):
            rhythm.set_intensity("extreme")


# ── UsageTracker ─────────────────────────────────────────────────────

class TestUsageTracker:
    def test_usage_rate_counts_only_decided(self):
        tracker = _usage.UsageTracker()
        a = tracker.record_prep("morning briefing")
        b = tracker.record_prep("research on X")
        c = tracker.record_prep("window prep")
        assert tracker.mark_used(a) is True
        assert tracker.mark_unused(b) is True
        rate = tracker.usage_rate()
        assert rate == {"total": 3, "decided": 2, "used": 1, "unused": 1,
                        "pending": 1, "rate": 0.5}
        # Pending item does not move the rate.
        assert tracker.usage_rate()["rate"] == 0.5

    def test_usage_rate_none_when_nothing_decided(self):
        tracker = _usage.UsageTracker()
        tracker.record_prep("briefing")
        assert tracker.usage_rate()["rate"] is None
        assert tracker.usage_rate()["decided"] == 0

    def test_double_verdict_refused(self):
        tracker = _usage.UsageTracker()
        item = tracker.record_prep("briefing")
        assert tracker.mark_used(item) is True
        assert tracker.mark_unused(item) is False
        assert tracker.mark_used("prep-nope") is False
        assert tracker.get(item)["status"] == "used"

    def test_usage_persists(self, tmp_path):
        path = tmp_path / "usage.json"
        tracker = _usage.UsageTracker(path)
        item = tracker.record_prep({"title": "briefing", "kind": "briefing"})
        tracker.mark_used(item)
        tracker2 = _usage.UsageTracker(path)
        assert tracker2.usage_rate()["rate"] == 1.0


# ── Preemption ───────────────────────────────────────────────────────

class TestPreemption:
    def test_ladder_order(self):
        assert _preemption.PRIORITY_LADDER == (
            "safety", "owner-direct", "scheduled", "ambient", "background")
        ladder = _preemption.PriorityLadder()
        assert ladder.rank("safety") < ladder.rank("background")
        assert ladder.higher("scheduled", "ambient") is True
        assert ladder.higher("ambient", "scheduled") is False

    def test_equal_priority_never_preempts(self):
        assert _preemption.should_preempt("ambient", "ambient") is False
        assert _preemption.should_preempt("scheduled", "ambient") is False
        assert _preemption.should_preempt("ambient", "scheduled") is True
        assert _preemption.should_preempt("background", "safety") is True

    def test_log_is_append_only_and_persists(self, tmp_path):
        path = tmp_path / "preemptions.jsonl"
        log = _preemption.PreemptionLog(path)
        rec = log.log_preempted("background-job", "safety-alert",
                                reason="safety wins",
                                victim_priority="background",
                                incoming_priority="safety")
        assert rec["victim"] == "background-job"
        assert rec["preempted_by"] == "safety-alert"
        log2 = _preemption.PreemptionLog(path)
        assert len(log2) == 1
        assert log2.for_victim("background-job")[0]["id"] == rec["id"]
        # Appending does not rewrite earlier entries.
        log2.log("other-job", "owner-direct")
        raw_lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(raw_lines) == 2
        assert json.loads(raw_lines[0])["victim"] == "background-job"

    def test_sort_highest_first_stable(self):
        ladder = _preemption.PriorityLadder()
        ordered = ladder.sort(["background", "safety", "ambient", "safety"])
        assert ordered == ["safety", "safety", "ambient", "background"]
