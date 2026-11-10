"""Doc 15 P6 — the §11 acceptance battery, composed end-to-end.

Each test is named for the doc's scenario it exercises (see
SCENARIO_TESTS at the bottom — the doc-wording → test map the
report carries). The battery composes the REAL shipped paths —
publish → decision → hold → release → ack; mission death → case →
projection → resolution — rather than re-implementing unit pins.
Scenarios whose full proof needs platform adapters or EC2 live
shapes (6, 13) exercise the slice this box can prove and the map
says so.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from nomorals.agents import inspection
from nomorals.agents.alert_policy import AlertPolicy
from nomorals.agents.notifier import STATE_HELD_QUIET, Notifier
from nomorals.agents.outcome_cases import (
    CaseError,
    StaleCaseGeneration,
)
from nomorals.agents.outcome_steward import (
    MissionsAdapter,
    OutcomeSteward,
    render_case_detail,
    run_stewardship_cycle,
)
from nomorals.core.jobs import JobRegistry
from nomorals.missions.mission import Mission, MissionStatus, MissionStore
from nomorals.storage.db import Database

# ── harness ───────────────────────────────────────────────────────────


class _SendResult:
    ok = True
    message_id = "m1"


class FakeGateway:
    def __init__(self, live=("telegram",)):
        self.live = set(live)
        self.sent: list[tuple] = []

    def status(self):
        return {p: {"running_in_session": p in self.live}
                for p in ("telegram", "whatsapp")}

    def send(self, platform, chat_ref, text, buttons=None):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult()


def _resolve(*args, **kwargs):
    if "default" in kwargs:
        return kwargs["default"]
    return args[-1] if args else None


def make_ctx(gateway=None):
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "nm.db"))
    db.migrate()
    partner = SimpleNamespace(
        owner_chats="telegram:111", proactive_enabled=True,
        proactive_briefing=True, proactive_watchers=True,
        quiet_start=0, quiet_end=0, timezone="UTC")
    settings = SimpleNamespace(workspace_dir=tmp, partner=partner,
                               resolve=_resolve)
    ctx = SimpleNamespace(
        db=db, settings=settings,
        extras={"gateway": gateway} if gateway else {})
    return ctx, db, tmp


def _mission(store, *, status, state=None, goal="do the thing",
             age_s=0.0):
    now = time.time() - age_s
    mission = Mission(goal=goal, status=status,
                      state=dict(state or {}),
                      created_at=now, updated_at=now)
    return store.create(mission)


def _liveness_state(extra=None):
    state = {"heartbeat": {"at": time.time() - 100000.0,
                           "pid": 999999},
             "last_error": ("runner died — no heartbeat for 27h "
                            "(pid 999999 is gone)")}
    state["status_note"] = f"reconciled: {state['last_error']}"
    state.update(extra or {})
    return state


def _presumed_dead(registry, fn):
    job = registry.submit(fn, tool="report", heartbeat_interval_s=1.0)
    for _ in range(100):
        job.last_heartbeat_at = time.time() - 100.0
        registry.reap()
        if job.status == "presumed-dead":
            break
        time.sleep(0.02)
    assert job.status == "presumed-dead"
    return job


_DEPLOY = {"subject": "deploy", "outcome": "succeeded",
           "prev_revision": "rev-1", "current_revision": "rev-2",
           "result_refs": ["artifact:build-9"], "impact": "v2 live"}


class NotificationScenarios(unittest.TestCase):
    """Doc §11 scenarios 1–6 + the digest-hold minimum."""

    def setUp(self):
        self.gw = FakeGateway()
        self.ctx, self.db, self.tmp = make_ctx(self.gw)
        self.notifier = Notifier(self.ctx, self.gw)

    # Scenario 1: unchanged monitoring ticks.
    def test_s01_100_ticks_zero_pushes_status_still_answers(self):
        for _ in range(100):
            self.notifier.publish(
                "jobwatch", "job tick", "still running",
                output_meta={"subject": "job:report",
                             "current_revision": "running"})
        self.assertEqual(self.gw.sent, [])          # at the gateway
        self.assertEqual(self.notifier.recent(10), [])
        coverage = AlertPolicy.candidate_coverage(self.db)
        self.assertEqual(coverage["suppressed"], 100)
        # An explicit status request still answers, completely.
        board = inspection.render_decisions(
            inspection.decision_entries(self.db, limit=5))
        self.assertIn("suppressed", board)

    # Scenario 2: duplicate completion events + restart.
    def test_s02_one_task_one_final_report_despite_duplicates(self):
        for _ in range(2):
            self.notifier.publish("deploy", "deploy finished",
                                  "v2 is live", output_meta=dict(_DEPLOY))
        # A restart: a second Notifier over the same db file.
        ctx2 = SimpleNamespace(
            db=self.db, settings=self.ctx.settings,
            extras={"gateway": self.gw})
        Notifier(ctx2, self.gw).publish(
            "deploy", "deploy finished", "v2 is live",
            output_meta=dict(_DEPLOY))
        self.assertEqual(len(self.gw.sent), 1)
        self.assertIn("deploy finished", self.gw.sent[0][2])

    # Scenario 3: identity is semantic, not title-shaped.
    def test_s03_same_title_twice_and_reworded_replay_once(self):
        meta_a = {"subject": "mission:render-a", "outcome": "succeeded",
                  "current_revision": "done",
                  "result_refs": ["artifact:a"]}
        meta_b = {"subject": "mission:render-b", "outcome": "succeeded",
                  "current_revision": "done",
                  "result_refs": ["artifact:b"]}
        self.notifier.publish("render", "Render completed", "a done",
                              output_meta=meta_a)
        self.notifier.publish("render", "Render completed", "b done",
                              output_meta=meta_b)
        self.assertEqual(len(self.gw.sent), 2)      # both arrive
        self.notifier.publish("render", "The render has completed",
                              "a done, reworded", output_meta=meta_a)
        self.assertEqual(len(self.gw.sent), 2)      # replay deduped

    # Scenario 4: new impact bypasses stale suppression.
    def test_s04_new_state_pages_then_its_ticks_suppress(self):
        for _ in range(3):
            self.notifier.publish(
                "jobwatch", "job tick", "running",
                output_meta={"subject": "job:x",
                             "current_revision": "running"})
        self.assertEqual(self.gw.sent, [])
        self.notifier.publish(
            "job", "job failed", "boom",
            output_meta={"subject": "job:x", "outcome": "failed",
                         "prev_revision": "running",
                         "current_revision": "failed",
                         "result_refs": ["log:run-7"]})
        self.assertEqual(len(self.gw.sent), 1)
        for _ in range(3):
            self.notifier.publish(
                "jobwatch", "job tick", "failed",
                output_meta={"subject": "job:x",
                             "current_revision": "failed"})
        self.assertEqual(len(self.gw.sent), 1)

    # Scenario 5: held backlog rechecks; obsolete progress dies.
    def test_s05_superseded_hold_never_replays_and_digest_folds(self):
        gw = FakeGateway(live=())                    # nothing delivers
        ctx, db, _tmp = make_ctx(gw)
        notifier = Notifier(ctx, gw)
        waiting = notifier.publish(
            "deploy", "deploy waiting", "queued",
            output_meta={"subject": "deploy",
                         "prev_revision": "rev-0",
                         "current_revision": "waiting",
                         "change_evidence": "deploy queued"})
        self.assertEqual(waiting["delivery_state"], "failed")
        db.execute("UPDATE notifications SET delivery_state = ? "
                   "WHERE id = ?", (STATE_HELD_QUIET, waiting["id"]))
        gw.live.add("telegram")
        notifier.publish("deploy", "deploy finished", "v2 live",
                         output_meta=dict(_DEPLOY))
        self.assertEqual(len(gw.sent), 1)            # completion only
        self.assertEqual(notifier.redeliver(), 0)    # hold superseded
        row = AlertPolicy.candidate_for_notification(db, waiting["id"])
        self.assertEqual(row["releases"][-1]["verdict"], "superseded")
        # Digest: a fresh item reaches the fold; once a completion
        # for the same subject lands, the stale item is retired at
        # the fold's recheck and the fold is empty — no digest
        # section is built, nothing stale is sent.
        item = notifier.publish(
            "job", "patrol waiting", "still waiting",
            alert_hints={"notice": True, "subject": "job:patrol"})
        self.assertEqual(item["tier"], "digest")
        self.assertEqual(
            [r["id"] for r in notifier.digest_unfolded()],
            [item["id"]])
        done = notifier.publish(
            "job", "patrol done", "finished",
            output_meta={"subject": "job:patrol",
                         "prev_revision": "r0",
                         "current_revision": "r1",
                         "outcome": "succeeded",
                         "result_refs": ["artifact:p1"]})
        self.assertTrue(done["delivered"])
        self.assertEqual(notifier.digest_unfolded(), [])
        item_row = AlertPolicy.candidate_for_notification(
            db, item["id"])
        self.assertEqual(item_row["releases"][-1]["verdict"],
                         "superseded")

    # The brief's digest-hold minimum (doc §3/§4 via P2, composed).
    def test_s05b_gapped_claim_digest_held_with_gaps_named(self):
        res = self.notifier.publish(
            "deploy", "deploy finished", "trust me",
            output_meta={"subject": "deploy",
                         "outcome": "succeeded"})
        self.assertEqual(res["tier"], "digest")
        self.assertIn("result_refs", res["material_gaps"])
        entries = inspection.decision_entries(self.db, limit=5)
        row = inspection.candidate_row_for(
            self.db, entries[0]["event_id"])
        text = inspection.render_why(row)
        self.assertIn("held for the morning digest", text)
        self.assertIn("result_refs", text)           # the gap, named

    # Scenario 6 (the slice this box proves): owner renders are
    # plain text — no markup dialect, no internal ids — on every
    # surface family.
    def test_s06_inspection_renders_are_plain_owner_text(self):
        self.notifier.publish("deploy", "deploy finished", "v2 live",
                              output_meta=dict(_DEPLOY))
        entries = inspection.decision_entries(self.db, limit=5)
        row = inspection.candidate_row_for(
            self.db, entries[0]["event_id"])
        for text in (inspection.render_why(row),
                     inspection.render_decisions(entries),
                     inspection.render_coverage(
                         AlertPolicy.candidate_coverage(self.db))):
            self.assertNotIn("<b>", text)
            self.assertNotIn("<i>", text)
            self.assertNotIn("oc-", text)
            self.assertNotIn(row["event_id"], text)


class StewardScenarios(unittest.TestCase):
    """Doc §11 scenarios 7–11."""

    def setUp(self):
        self.ctx, self.db, self.tmp = make_ctx()
        self.missions = MissionStore(self.db)
        self.registry = JobRegistry()
        self.steward = OutcomeSteward(self.db, registry=self.registry)
        self.cases = self.steward.cases

    # Scenario 7a: a slow, healthy operation is left alone.
    def test_s07a_healthy_running_mission_opens_nothing(self):
        _mission(self.missions, status=MissionStatus.RUNNING,
                 state={"heartbeat": {"at": time.time(), "pid": 4242}},
                 goal="slow render")
        report = self.steward.sweep()
        self.assertEqual(report["opened"], 0)

    # Scenario 7b: stalled-but-alive work opens a case.
    def test_s07b_stalled_mission_opens_a_case(self):
        _mission(self.missions, status=MissionStatus.PENDING,
                 goal="queued forever", age_s=90000.0)
        self.steward.sweep()
        case = self.cases.list_cases()[0]
        self.assertEqual(case.domain, "missions")
        self.assertEqual(case.verdict, "outcome_unknown")

    # Scenario 7c + the brief's minimum: dead runner, remote effect
    # succeeded → NOT a total failure; resolves from read-back.
    def test_s07c_dead_runner_evidenced_effect_resolves(self):
        mission = _mission(
            self.missions, status=MissionStatus.FAILED,
            state=_liveness_state({
                "plan": {"steps": [
                    {"name": "only", "goal": "g", "role": "scout"}]},
                "completed_steps": ["only"],
                "outputs": {"only": {"ok": True}}}),
            goal="landed work")
        # Discovery itself sees the effect: no unknown case opens.
        self.assertEqual(self.steward.sweep()["opened"], 0)
        # A case opened anyway resolves citing the projection.
        case, created = self.cases.open_case(
            domain="missions", kind="mission", subject=mission.goal,
            subject_key=f"mission:{mission.id}",
            reason="unknown-outcome",
            detail={"mission_id": mission.id})
        self.assertTrue(created)
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        closed = self.cases.get(case.id)
        self.assertEqual(closed.verdict, "succeeded")
        cited = self.cases.get_receipt(closed.evidence_receipt_id)
        self.assertEqual(cited.source, "missions.effect_outcome")

    # Scenario 7d + brief minimum: presumed-dead job, late result.
    def test_s07d_presumed_dead_job_late_result_resolves(self):
        gate = threading.Event()

        def work():
            gate.wait(10)
            return {"report": "landed late"}

        job = _presumed_dead(self.registry, work)
        self.steward.sweep()
        case = self.cases.list_cases()[0]
        gate.set()
        for _ in range(100):
            if job.late_result:
                break
            time.sleep(0.05)
        self.assertTrue(job.late_result)
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        self.assertEqual(self.cases.get(case.id).verdict, "succeeded")

    # Scenario 8: orphans + uncollected results, verified + recorded.
    def test_s08_orphans_and_uncollected_verified_and_recorded(self):
        job = self.registry.submit(lambda: {"report": "landed"},
                                   tool="report")
        self.registry.collect(job.id, timeout=10)
        job.ended_at = time.time() - 700.0
        _mission(self.missions, status=MissionStatus.RUNNING,
                 state={"heartbeat": {"at": time.time() - 100000.0,
                                      "pid": 999999}},
                 goal="orphaned render")
        report = run_stewardship_cycle(self.db,
                                       registry=self.registry)
        self.assertEqual(report["counts"]["resolved"], 1)
        self.assertEqual(report["counts"]["closed-unknown"], 1)
        by_domain = {c.domain: c for c in self.cases.list_cases(
            limit=10)}
        job_case = by_domain["jobs"]
        self.assertEqual(job_case.verdict, "succeeded")
        self.assertIsNotNone(job_case.evidence_receipt_id)
        # The inspection render shows outcome + evidence trail.
        text = render_case_detail(
            job_case, self.cases.receipts(job_case.id))
        self.assertIn("succeeded", text)
        self.assertIn("evidence trail:", text)
        self.assertNotIn(job_case.id, text)

    # Scenario 9 + brief minimum: unknown stays unknown; no blind
    # retry; a later safe read-back resolves with NEW evidence.
    def test_s09_unknown_stays_unknown_then_readback_resolves(self):
        gate = threading.Event()
        job = _presumed_dead(self.registry, lambda: gate.wait(30))
        self.steward.sweep()
        case = self.cases.list_cases()[0]
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "closed-unknown")
        closed = self.cases.get(case.id)
        self.assertEqual(closed.verdict, "outcome_unknown")
        receipts_before = len(self.cases.receipts(case.id))
        # No blind retry anywhere: the surface grammar has no
        # retry verb, and the domain's own evidence — carried
        # faithfully into the render — counsels against re-issuing.
        text = render_case_detail(closed,
                                  self.cases.receipts(case.id))
        self.assertIn("outcome unknown", text)
        self.assertIn("never re-issue blindly", text)
        self.assertNotIn("/cases retry", text)
        # The result lands late; re-discovery reopens the case and a
        # fresh run resolves it on a NEW evidence receipt.
        gate.set()
        for _ in range(100):
            if job.late_result:
                break
            time.sleep(0.05)
        self.assertTrue(job.late_result)
        self.steward.sweep()
        reopened = self.cases.get(case.id)
        self.assertEqual(reopened.state, "open")
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        final = self.cases.get(case.id)
        self.assertEqual(final.verdict, "succeeded")
        self.assertGreater(len(self.cases.receipts(case.id)),
                           receipts_before)
        cited = self.cases.get_receipt(final.evidence_receipt_id)
        self.assertEqual(cited.generation, final.generation)

    # Scenario 10: concurrency fencing on a discovered case.
    def test_s10_stale_generation_cannot_overwrite(self):
        _mission(self.missions, status=MissionStatus.FAILED,
                 state=_liveness_state(), goal="mystery work")
        self.steward.sweep()
        case = self.cases.list_cases()[0]
        _c, gen1 = self.cases.claim(case.id)
        case, gen2 = self.cases.claim(case.id)
        evidence = self.cases.add_receipt(
            case.id, kind="evidence", source="missions.store",
            note="fresh read-back", generation=gen2)
        with self.assertRaises(StaleCaseGeneration):
            self.cases.resolve(case.id, verdict="failed",
                               evidence_receipt_id=evidence.id,
                               generation=gen1)
        # State untouched by the stale write.
        self.assertEqual(self.cases.get(case.id).state,
                         "investigating")
        with self.assertRaises(CaseError):
            self.cases.resolve(case.id, verdict="outcome_unknown",
                               evidence_receipt_id=evidence.id,
                               generation=gen2)
        closed = self.cases.resolve(
            case.id, verdict="failed",
            evidence_receipt_id=evidence.id, generation=gen2)
        self.assertEqual(closed.verdict, "failed")

    # Scenario 11: a crashed investigation recovers — no recursion,
    # no duplicate effects.
    def test_s11_verifier_crash_recovers_exactly_once(self):
        _mission(self.missions, status=MissionStatus.FAILED,
                 state=_liveness_state(), goal="crashy work")
        self.steward.sweep()
        case = self.cases.list_cases()[0]

        class FlakyAdapter(MissionsAdapter):
            def __init__(self, db):
                super().__init__(db)
                self.calls = 0

            def collect(self, case, candid=None):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("verifier host died")
                return super().collect(case, candid)

        flaky = FlakyAdapter(self.db)
        steward = OutcomeSteward(
            self.db, cases=self.cases, adapters=[flaky],
            registry=self.registry)
        first = steward.run_case(case.id)
        self.assertEqual(first["outcome"], "incomplete")
        mid = self.cases.get(case.id)
        self.assertEqual(mid.state, "investigating")
        self.assertEqual(mid.verdict, "outcome_unknown")
        terminal = [r for r in self.cases.receipts(case.id)
                    if r.kind in ("resolution", "closed-unknown")]
        self.assertEqual(terminal, [])
        second = steward.run_case(case.id)
        self.assertEqual(second["outcome"], "closed-unknown")
        terminal = [r for r in self.cases.receipts(case.id)
                    if r.kind in ("resolution", "closed-unknown")]
        self.assertEqual(len(terminal), 1)
        # No recursion: the steward's own completed investigation
        # contract never becomes a case, and the next full cycle
        # terminates with the SAME single case (re-discovery may
        # reopen it — the mission is still dead — but no new case
        # is minted and nothing loops).
        run_stewardship_cycle(self.db, registry=self.registry)
        all_cases = self.cases.list_cases(limit=10)
        self.assertEqual(len(all_cases), 1)
        self.assertEqual(all_cases[0].domain, "missions")


class SeparationScenario(unittest.TestCase):
    """Doc §11 scenario 12: operation success and notification
    failure are separate facts with independent retries."""

    def test_s12_operation_success_notification_failure_separate(self):
        gw = FakeGateway(live=())
        ctx, db, _tmp = make_ctx(gw)
        missions = MissionStore(db)
        mission = _mission(missions, status=MissionStatus.DONE,
                           goal="landed work")
        notifier = Notifier(ctx, gw)
        res = notifier.publish(
            "mission", "mission done", "landed",
            output_meta={"subject": "mission:landed",
                         "outcome": "succeeded",
                         "current_revision": "done",
                         "result_refs": ["mission-record"]})
        self.assertEqual(res["delivery_state"], "failed")
        # The operation's own record is untouched by the failure.
        self.assertEqual(missions.get(mission.id).status,
                         MissionStatus.DONE)
        updated_at = missions.get(mission.id).updated_at
        gw.live.add("telegram")
        self.assertEqual(notifier.redeliver(), 1)
        self.assertEqual(len(gw.sent), 1)
        # Still untouched — delivery retried, the operation was not
        # re-run.
        after = missions.get(mission.id)
        self.assertEqual(after.status, MissionStatus.DONE)
        self.assertEqual(after.updated_at, updated_at)


#: The doc §11 map — every scenario names its test(s). Scenarios 6
#: and 13 are partial on this box by their nature (platform adapters
#: / EC2 live shapes; the full storage matrix incl. Mongo belongs
#: to the Mongo programme's suites) — the report carries the
#: verdicts; this dict pins that every scenario is accounted for.
SCENARIO_TESTS = {
    1: ["NotificationScenarios.test_s01_100_ticks_zero_pushes_status_still_answers"],
    2: ["NotificationScenarios.test_s02_one_task_one_final_report_despite_duplicates"],
    3: ["NotificationScenarios.test_s03_same_title_twice_and_reworded_replay_once"],
    4: ["NotificationScenarios.test_s04_new_state_pages_then_its_ticks_suppress"],
    5: ["NotificationScenarios.test_s05_superseded_hold_never_replays_and_digest_folds",
        "NotificationScenarios.test_s05b_gapped_claim_digest_held_with_gaps_named"],
    6: ["NotificationScenarios.test_s06_inspection_renders_are_plain_owner_text"],
    7: ["StewardScenarios.test_s07a_healthy_running_mission_opens_nothing",
        "StewardScenarios.test_s07b_stalled_mission_opens_a_case",
        "StewardScenarios.test_s07c_dead_runner_evidenced_effect_resolves",
        "StewardScenarios.test_s07d_presumed_dead_job_late_result_resolves"],
    8: ["StewardScenarios.test_s08_orphans_and_uncollected_verified_and_recorded"],
    9: ["StewardScenarios.test_s09_unknown_stays_unknown_then_readback_resolves"],
    10: ["StewardScenarios.test_s10_stale_generation_cannot_overwrite"],
    11: ["StewardScenarios.test_s11_verifier_crash_recovers_exactly_once"],
    12: ["SeparationScenario.test_s12_operation_success_notification_failure_separate"],
    13: ["ScenarioMapTests.test_s13_every_scenario_is_accounted_for"],
}


class ScenarioMapTests(unittest.TestCase):
    def test_s13_every_scenario_is_accounted_for(self):
        import sys

        module = sys.modules[__name__]
        self.assertEqual(sorted(SCENARIO_TESTS), list(range(1, 14)))
        for scenario, names in SCENARIO_TESTS.items():
            for name in names:
                klass, _, method = name.partition(".")
                self.assertTrue(
                    hasattr(getattr(module, klass), method),
                    f"scenario {scenario}: {name} missing")


if __name__ == "__main__":
    unittest.main()
