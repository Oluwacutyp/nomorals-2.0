"""Acceptance tests for Prompt 01 — the self-improvement engine.

Covers the spec's contract: skill implicated in >=3 failures gets a gated
proposal; a bad edit is reverted and fingerprinted; lessons inject through
the choke point; high-usefulness lessons surface as promotion candidates;
repeated tool patterns synthesize a registered, smoke-tested skill; a worse
canary auto-reverts with numbers; `nm improve status` works; the exclusion
list refuses vault/connector-auth edits.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.storage.db import Database
from nomorals.agents.failure import (
    FailureAnalyzer, FailureCase, enrich_with_lessons, failure_fingerprint)
from nomorals.agents.skill_evolution import (
    SkillEvolutionLoop, SkillEvolutionError, apply_unified_diff,
    count_changed_lines, diff_fingerprint, ensure_improvement_schedule)
from nomorals.agents.skill_canary import CanaryRollout, decide
from nomorals.agents.skill_synthesis import (
    SkillSynthesizer, detect_patterns, smoke_test_skill)


def _ctx(mode="approval"):
    db = Database(":memory:")
    db.migrate()  # Database() does not auto-migrate; the real CLI calls it
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(
            improvement=SimpleNamespace(mode=mode)),
        router=None)


_GOOD_DIFF = """--- a/skill
+++ b/skill
@@ -1,3 +1,4 @@
 line one
-old line
+new line
+added line
 line three
"""

_BAD_DIFF = """--- a/skill
+++ b/skill
@@ -1,3 +1,4 @@
-line one
+---
+title: broken
 old line
 line three
"""


def _fake_proposer(diff):
    def _propose(skill_name, current_text, failures):
        return diff
    return _propose


def _skill_body():
    return "line one\nold line\nline three\n"


class DiffUtilsTestCase(unittest.TestCase):
    def test_apply_unified_diff(self):
        out = apply_unified_diff("line one\nold line\nline three\n",
                                 _GOOD_DIFF)
        self.assertEqual(out, "line one\nnew line\nadded line\nline three\n")

    def test_apply_rejects_context_mismatch(self):
        with self.assertRaises(SkillEvolutionError):
            apply_unified_diff("totally different\ntext\n", _GOOD_DIFF)

    def test_count_changed_lines(self):
        self.assertEqual(count_changed_lines(_GOOD_DIFF), 3)
        self.assertEqual(count_changed_lines("no diff here"), 0)

    def test_fingerprint_stable(self):
        self.assertEqual(diff_fingerprint(_GOOD_DIFF),
                         diff_fingerprint(_GOOD_DIFF + "\n"))
        self.assertNotEqual(diff_fingerprint(_GOOD_DIFF),
                            diff_fingerprint(_BAD_DIFF))

    def test_fingerprint_is_deterministic(self):
        fp1 = failure_fingerprint("tool", "TimeoutError: timed out after 30s",
                                  "fetch", "web")
        fp2 = failure_fingerprint("tool", "TimeoutError: timed out after 120s",
                                  "fetch", "web")
        self.assertEqual(fp1, fp2)  # numbers normalize away
        fp3 = failure_fingerprint("tool", "KeyError: 'missing'", "fetch",
                                  "web")
        self.assertNotEqual(fp1, fp3)


class DetectTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()

    def test_detect_triggers_at_threshold(self):
        analyzer = FailureAnalyzer(self.ctx)
        for i in range(3):
            analyzer.record("tool", f"search failed {i}",
                            "ConnectionError: timed out", skill="web_search")
        for i in range(2):
            analyzer.record("tool", f"other failed {i}",
                            "ConnectionError: timed out", skill="fetch_page")
        loop = SkillEvolutionLoop(self.ctx)
        cands = loop.detect()
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["skill"], "web_search")
        self.assertEqual(cands[0]["count"], 3)

    def test_detect_window_excludes_old_failures(self):
        analyzer = FailureAnalyzer(self.ctx)
        for i in range(3):
            analyzer.record("tool", f"old fail {i}",
                            "ConnectionError: timed out", skill="web_search")
        # backdate them out of the window
        self.ctx.db.execute(
            "UPDATE failures SET ts=? WHERE skill=?",
            (time.time() - 30 * 86400, "web_search"))
        loop = SkillEvolutionLoop(self.ctx, window_days=7)
        self.assertEqual(loop.detect(), [])


class GateTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx(mode="approval")
        from nomorals.agents.skills import SkillLibrary
        self.lib = SkillLibrary(self.ctx.db)
        self.skill = self.lib.save(
            "test skill alpha", kind="strategy", body=_skill_body(),
            description="a test skill", tags=["test"], source="test")

    def _loop(self, diff, **kw):
        return SkillEvolutionLoop(
            self.ctx, proposer=_fake_proposer(diff),
            test_runner=lambda cmd: (0, "ok"),
            benchmark_fn=lambda s, f: {"ok": True, "skipped": True,
                                       "detail": "fake"},
            **kw)

    def test_good_edit_passes_gate_and_stages_in_approval(self):
        loop = self._loop(_GOOD_DIFF)
        proposal = loop.propose("test skill alpha")
        self.assertEqual(proposal["changed_lines"], 3)
        passed, results = loop.gate(proposal)
        self.assertTrue(passed, results)
        self.assertTrue(results["static"]["ok"])
        self.assertTrue(results["tests"]["ok"])
        rec = loop.apply_proposal(proposal)
        self.assertEqual(rec.status, "staged")
        self.assertEqual(rec.mode, "approval")
        # staged edits do not change the live skill
        self.assertEqual(
            self.lib.get_by_name("test skill alpha").body, _skill_body())

    def test_bad_edit_reverted_and_fingerprinted(self):
        loop = self._loop(_BAD_DIFF)
        proposal = loop.propose("test skill alpha")
        passed, results = loop.gate(proposal)
        self.assertFalse(passed)
        self.assertFalse(results["static"]["ok"])
        rec = loop.apply_proposal(proposal)
        self.assertEqual(rec.status, "reverted")
        # the same diff is never proposed again
        rec2 = loop.apply_proposal(proposal)
        self.assertEqual(rec2.status, "refused")
        self.assertIn("already regressed",
                      json.dumps(rec2.gate_results))

    def test_oversized_diff_rejected_in_autonomous(self):
        big = ("--- a/s\n+++ b/s\n@@ -1,1 +1,100 @@\n-x\n"
               + "".join(f"+line{i}\n" for i in range(70)))
        ctx = _ctx(mode="autonomous")
        from nomorals.agents.skills import SkillLibrary
        SkillLibrary(ctx.db).save("test skill alpha", kind="strategy",
                                  body=_skill_body(), source="test")
        loop = SkillEvolutionLoop(
            ctx, proposer=_fake_proposer(big),
            test_runner=lambda cmd: (0, "ok"),
            benchmark_fn=lambda s, f: {"ok": True, "skipped": True})
        with self.assertRaises(SkillEvolutionError) as cm:
            loop.propose("test skill alpha")
        self.assertIn("approval mode", str(cm.exception))

    def test_autonomous_applies_and_records(self):
        ctx = _ctx(mode="autonomous")
        from nomorals.agents.skills import SkillLibrary
        SkillLibrary(ctx.db).save("test skill alpha", kind="strategy",
                                  body=_skill_body(), source="test")
        loop = SkillEvolutionLoop(
            ctx, proposer=_fake_proposer(_GOOD_DIFF),
            test_runner=lambda cmd: (0, "ok"),
            benchmark_fn=lambda s, f: {"ok": True, "skipped": True,
                                       "detail": "fake"})
        # canary import works in-repo; force direct apply by making the
        # target look like a file is not possible for db skills — instead
        # verify the full autonomous path including canary staging
        proposal = loop.propose("test skill alpha")
        rec = loop.apply_proposal(proposal)
        # db skill in autonomous -> canary run, live body unchanged
        self.assertIn(rec.status, ("canary", "applied"))
        self.assertTrue(rec.gate_results.get("static", {}).get("ok"))

    def test_manual_revert_restores_body(self):
        loop = self._loop(_GOOD_DIFF)
        proposal = loop.propose("test skill alpha")
        # simulate an applied edit
        rec = SkillEditRecord_for_test(loop, proposal, status="applied")
        self.ctx.db.execute(
            "UPDATE skills SET body=? WHERE id=?",
            (proposal["new_text"], self.skill.id))
        out = loop.revert_edit(rec["id"])
        self.assertTrue(out["ok"])
        self.assertEqual(
            self.lib.get_by_name("test skill alpha").body, _skill_body())

    def test_tick_promotes_high_usefulness_lesson_into_skill(self):
        from nomorals.agents.skills import SkillLibrary
        lib = SkillLibrary(self.ctx.db)
        skill = lib.save("promoted skill", kind="strategy",
                         body=_skill_body(), source="test")
        analyzer = FailureAnalyzer(self.ctx)
        lesson = analyzer.learn_from_failure(
            FailureCase(source="tool", summary="x", error="KeyError: 'a'"),
            analysis={"category": "c", "root_cause": "r", "lesson": "l",
                      "fix": "f", "prevention": "p"})
        self.ctx.db.execute(
            "UPDATE lessons SET skill_id=?, times_surfaced=15, "
            "times_prevented=14 WHERE id=?", (skill.id, lesson.id))
        loop = SkillEvolutionLoop(
            self.ctx, proposer=_fake_proposer(_GOOD_DIFF),
            test_runner=lambda cmd: (0, "ok"),
            benchmark_fn=lambda s, f: {"ok": True, "skipped": True})
        out = loop.tick()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].skill_name, "promoted skill")
        self.assertEqual(out[0].status, "staged")  # approval mode


def SkillEditRecord_for_test(loop, proposal, status="applied"):
    """Store a fake applied edit record; returns the row dict."""
    from nomorals.agents.skill_evolution import SkillEditRecord
    rec = SkillEditRecord(
        id="skedit-test1", skill_name=proposal["skill_name"],
        target_kind=proposal["target_kind"], target_ref=proposal["target_ref"],
        before_hash=proposal["before_hash"], after_hash=proposal["after_hash"],
        diff=proposal["diff"], mode="autonomous", status=status,
        fingerprint=proposal["fingerprint"], created_at=time.time(),
        decided_at=time.time())
    loop._store(rec)
    return {"id": rec.id}


class ExclusionTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        from nomorals.agents.skills import SkillLibrary
        self.lib = SkillLibrary(self.ctx.db)

    def test_vault_named_skill_refused(self):
        self.lib.save("my vault helper", kind="strategy", body="x = 1\n",
                      source="test")
        loop = SkillEvolutionLoop(self.ctx, proposer=_fake_proposer(_GOOD_DIFF))
        with self.assertRaises(SkillEvolutionError) as cm:
            loop.propose("my vault helper")
        self.assertIn("refused", str(cm.exception))

    def test_connector_path_in_diff_refused(self):
        self.lib.save("plain skill", kind="strategy",
                      body="line one\nold line\n", source="test")
        evil = ("--- a/s\n+++ b/s\n@@ -1,2 +1,2 @@\n line one\n-old line\n"
                "+import nomorals/connectors/thing\n")
        loop = SkillEvolutionLoop(self.ctx, proposer=_fake_proposer(evil))
        with self.assertRaises(SkillEvolutionError) as cm:
            loop.propose("plain skill")
        self.assertIn("refused", str(cm.exception))


class LessonV2TestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()

    def test_choke_point_injects_lessons(self):
        analyzer = FailureAnalyzer(self.ctx)
        case = FailureCase(source="tool", summary="fetch broke",
                           error="KeyError: 'url'")
        lesson = analyzer.learn_from_failure(
            case, analysis={"category": "missing_key",
                            "root_cause": "no url key",
                            "lesson": "always pass url",
                            "fix": "pass url",
                            "prevention": "always include the url argument"})
        # the choke point: a planner call with no explicit call site
        block = enrich_with_lessons(self.ctx, "fetch the url please")
        self.assertIn("always include the url argument", block)
        # surfacing was counted
        row = self.ctx.db.query_one(
            "SELECT times_surfaced FROM lessons WHERE id=?", (lesson.id,))
        self.assertEqual(row["times_surfaced"], 1)
        surf = self.ctx.db.query_one(
            "SELECT COUNT(*) AS n FROM lesson_surfacings WHERE lesson_id=?",
            (lesson.id,))
        self.assertEqual(surf["n"], 1)

    def test_usefulness_prevented_when_no_recurrence(self):
        analyzer = FailureAnalyzer(self.ctx)
        case = FailureCase(source="tool", summary="x", error="KeyError: 'a'")
        lesson = analyzer.learn_from_failure(
            case, analysis={"category": "missing_key", "root_cause": "r",
                            "lesson": "l", "fix": "f",
                            "prevention": "p always include url"})
        enrich_with_lessons(self.ctx, "include url here")
        # age the surfacing past the window with no recurrence
        self.ctx.db.execute(
            "UPDATE lesson_surfacings SET surfaced_at=?",
            (time.time() - 10 * 86400,))
        out = analyzer.evaluate_usefulness(window_days=7)
        self.assertEqual(out["evaluated"], 1)
        row = self.ctx.db.query_one(
            "SELECT times_prevented FROM lessons WHERE id=?", (lesson.id,))
        self.assertEqual(row["times_prevented"], 1)

    def test_recurrence_not_counted_as_prevented(self):
        analyzer = FailureAnalyzer(self.ctx)
        case = FailureCase(source="tool", summary="x",
                           error="KeyError: 'a'", operation="op1")
        lesson = analyzer.learn_from_failure(
            case, analysis={"category": "missing_key", "root_cause": "r",
                            "lesson": "l", "fix": "f", "prevention": "p"})
        enrich_with_lessons(self.ctx, "something")
        surf_at = time.time() - 10 * 86400
        self.ctx.db.execute("UPDATE lesson_surfacings SET surfaced_at=?",
                            (surf_at,))
        # a repeat failure with the same fingerprint AFTER surfacing
        fp = failure_fingerprint("tool", "KeyError: 'a'", "op1", "")
        self.ctx.db.execute(
            "INSERT INTO failures (id, source, summary, error, family, "
            "lesson, ts, fingerprint, skill) VALUES "
            "('f1','tool','x','KeyError: ''a''','missing_key','',?,?,'')",
            (surf_at + 100, fp))
        out = analyzer.evaluate_usefulness(window_days=7)
        self.assertEqual(out["evaluated"], 1)
        row = self.ctx.db.query_one(
            "SELECT times_prevented FROM lessons WHERE id=?", (lesson.id,))
        self.assertEqual(row["times_prevented"], 0)

    def test_low_usefulness_demoted(self):
        analyzer = FailureAnalyzer(self.ctx)
        lesson = analyzer.learn_from_failure(
            FailureCase(source="tool", summary="x", error="KeyError: 'a'"),
            analysis={"category": "c", "root_cause": "r", "lesson": "l",
                      "fix": "f", "prevention": "p"})
        self.ctx.db.execute(
            "UPDATE lessons SET times_surfaced=10, times_prevented=1 "
            "WHERE id=?", (lesson.id,))
        out = analyzer.evaluate_usefulness()
        self.assertEqual(out["demoted"], 1)
        row = self.ctx.db.query_one(
            "SELECT demoted FROM lessons WHERE id=?", (lesson.id,))
        self.assertEqual(row["demoted"], 1)

    def test_zero_usefulness_archived_not_deleted(self):
        analyzer = FailureAnalyzer(self.ctx)
        lesson = analyzer.learn_from_failure(
            FailureCase(source="tool", summary="x", error="KeyError: 'a'"),
            analysis={"category": "c", "root_cause": "r", "lesson": "l",
                      "fix": "f", "prevention": "p"})
        self.ctx.db.execute(
            "UPDATE lessons SET times_surfaced=20, times_prevented=0 "
            "WHERE id=?", (lesson.id,))
        out = analyzer.evaluate_usefulness()
        self.assertEqual(out["archived"], 1)
        gone = self.ctx.db.query_one("SELECT 1 FROM lessons WHERE id=?",
                                     (lesson.id,))
        self.assertIsNone(gone)
        kept = self.ctx.db.query_one(
            "SELECT archive_reason FROM lessons_archive WHERE id=?",
            (lesson.id,))
        self.assertIsNotNone(kept)

    def test_promotion_candidates(self):
        analyzer = FailureAnalyzer(self.ctx)
        lesson = analyzer.learn_from_failure(
            FailureCase(source="tool", summary="x", error="KeyError: 'a'"),
            analysis={"category": "c", "root_cause": "r", "lesson": "l",
                      "fix": "f", "prevention": "p"})
        self.ctx.db.execute(
            "UPDATE lessons SET skill_id='skill1', times_surfaced=15, "
            "times_prevented=14 WHERE id=?", (lesson.id,))
        cands = analyzer.promotion_candidates()
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].id, lesson.id)
        self.assertGreaterEqual(cands[0].usefulness, 0.8)


class SynthesisTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx(mode="autonomous")

    def _seed_traces(self, n=6):
        now = time.time()
        seq = ["web_search", "fetch_page", "summarize"]
        k = 0
        for rep in range(n):
            for tool in seq:
                self.ctx.db.execute(
                    "INSERT INTO tool_calls (id, actor, tool, capability, "
                    "decision, args_digest, status, duration_ms, "
                    "result_digest, error, created_at) VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?)",
                    (f"tc{k}", "devon", tool, "", "allow", "", "ok",
                     10.0, "", "", now - (n - rep) * 100 + k))
                k += 1

    def test_detect_repeated_pattern(self):
        self._seed_traces()
        patterns = detect_patterns(self.ctx.db, min_repeats=5)
        triples = [p for p in patterns
                   if p.tools == ["web_search", "fetch_page", "summarize"]]
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].count, 6)

    def test_synthesize_registers_and_passes_smoke(self):
        self._seed_traces()
        synth = SkillSynthesizer(self.ctx)
        patterns = detect_patterns(self.ctx.db, min_repeats=5)
        triple = next(p for p in patterns
                      if p.tools == ["web_search", "fetch_page", "summarize"])
        out = synth.synthesize_pattern(triple)
        self.assertTrue(out["ok"], out)
        self.assertFalse(out["staged"])  # autonomous: probation, not staged
        skill = synth.skills.get_by_name(out["name"])
        self.assertIsNotNone(skill)
        self.assertIn("probation", skill.tags)
        # smoke test replays a source trace through the skill
        ok, detail = smoke_test_skill(
            skill.body, ["web_search", "fetch_page", "summarize"])
        self.assertTrue(ok, detail)
        # provenance recorded
        row = self.ctx.db.query_one(
            "SELECT value FROM kv_store WHERE key=?",
            (f"skill.provenance.{skill.id}",))
        self.assertIsNotNone(row)
        prov = json.loads(row["value"])
        self.assertIn("synthesized_from", prov)
        self.assertIn("synthesized_at", prov)

    def test_covered_pattern_not_synthesized(self):
        self._seed_traces()
        synth = SkillSynthesizer(self.ctx)
        synth.skills.save("web search fetch page summarize helper",
                          kind="workflow", body="does web_search fetch_page",
                          description="covers web_search fetch_page summarize",
                          tags=["web_search", "fetch_page", "summarize"],
                          source="test")
        patterns = detect_patterns(self.ctx.db, min_repeats=5)
        triples = [p for p in patterns
                   if p.tools == ["web_search", "fetch_page", "summarize"]]
        self.assertEqual(triples, [])

    def test_probation_review_removes_unused(self):
        self._seed_traces()
        synth = SkillSynthesizer(self.ctx)
        patterns = detect_patterns(self.ctx.db, min_repeats=5)
        triple = next(p for p in patterns
                      if p.tools == ["web_search", "fetch_page", "summarize"])
        out = synth.synthesize_pattern(triple)
        skill_id = out["skill_id"]
        # expire the probation
        self.ctx.db.execute(
            "UPDATE kv_store SET value=? WHERE key=?",
            (json.dumps({"until": time.time() - 1, "pattern": {}}),
             f"skill.probation.{skill_id}"))
        reviewed = synth.review_probation()
        self.assertEqual(len(reviewed), 1)
        self.assertFalse(reviewed[0]["kept"])
        self.assertIsNone(synth.skills.get(skill_id))


class CanaryTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx(mode="autonomous")

    def test_decide_pure(self):
        self.assertEqual(decide(19, 20, 18, 20), "promote")
        self.assertEqual(decide(10, 20, 19, 20), "revert")
        self.assertEqual(decide(0, 0, 19, 20), "inconclusive")

    def test_worse_canary_auto_reverts_with_numbers(self):
        rollout = CanaryRollout(self.ctx, min_sample=4, min_hours=0)
        run = rollout.start("test skill alpha", "new body here",
                            baseline_body="old body")
        self.assertTrue(run["ok"])
        self.assertNotEqual(run["canary_hash"], run["baseline_hash"])
        # canary: 1/4 ok; baseline: 4/4 ok
        for ok_ in (True, False, False, False):
            rollout.observe("test skill alpha", run["canary_hash"], ok_)
        for _ in range(4):
            rollout.observe("test skill alpha", run["baseline_hash"], True)
        out = rollout.evaluate("test skill alpha")
        self.assertTrue(out["ok"])
        self.assertEqual(out["decision"], "revert")
        self.assertEqual(out["status"], "reverted")
        self.assertEqual(out["canary"], {"ok": 1, "n": 4, "rate": 0.25})
        self.assertEqual(out["baseline"], {"ok": 4, "n": 4, "rate": 1.0})
        # run row records the decision + numbers
        row = self.ctx.db.query_one(
            "SELECT status, decision, decision_detail FROM canary_runs "
            "WHERE id=?", (run["id"],))
        self.assertEqual(row["status"], "reverted")
        self.assertEqual(row["decision"], "revert")
        detail = json.loads(row["decision_detail"])
        self.assertEqual(detail["canary"]["rate"], 0.25)

    def test_better_canary_promotes(self):
        rollout = CanaryRollout(self.ctx, min_sample=4, min_hours=0)
        run = rollout.start("s", "new", baseline_body="old")
        for _ in range(4):
            rollout.observe("s", run["canary_hash"], True)
            rollout.observe("s", run["baseline_hash"], False)
        out = rollout.evaluate("s")
        self.assertEqual(out["decision"], "promote")
        self.assertEqual(out["status"], "promoted")

    def test_evaluate_waits_for_sample(self):
        rollout = CanaryRollout(self.ctx, min_sample=20, min_hours=9999)
        run = rollout.start("s", "new", baseline_body="old")
        rollout.observe("s", run["canary_hash"], True)
        out = rollout.evaluate("s")
        self.assertEqual(out["decision"], "waiting")

    def test_version_history_and_restore(self):
        from nomorals.agents.skills import SkillLibrary
        lib = SkillLibrary(self.ctx.db)
        skill = lib.save("restorable", kind="strategy", body="v1",
                         source="test")
        rollout = CanaryRollout(self.ctx)
        h1 = rollout.record_version("restorable", "v1", source="test")
        h2 = rollout.record_version("restorable", "v2", source="test")
        self.assertNotEqual(h1, h2)
        vers = rollout.versions("restorable")
        self.assertEqual(len(vers), 2)
        self.assertEqual(vers[0]["parent_hash"], vers[1]["version_hash"])
        out = rollout.restore_version("restorable", h1)
        self.assertTrue(out["ok"])
        self.assertEqual(lib.get_by_name("restorable").body, "v1")

    def test_routing_seam(self):
        import random
        rollout = CanaryRollout(self.ctx, rng=random.Random(0))
        run = rollout.start("s", "CANARY", baseline_body="BASE",
                            fraction=1.0)
        body, ver = rollout.resolve_body("s", "BASE")
        self.assertEqual((body, ver), ("CANARY", "canary"))


class OrchestratorInjectionTestCase(unittest.TestCase):
    """A planner with no previous lesson call site (MasterOrchestrator.plan)
    now receives lessons through the choke point."""

    def setUp(self):
        self.ctx = _ctx()
        self.captured = []

    def _orch(self):
        from nomorals.agents.orchestrator import MasterOrchestrator

        captured = self.captured

        class FakeRouter:
            def chat(self, messages, params):
                captured.append(messages[0].content
                                if hasattr(messages[0], "content")
                                else str(messages[0]))
                return SimpleNamespace(ok=False, model="fake", text="")

        ctx = SimpleNamespace(db=self.ctx.db, router=FakeRouter(),
                              settings=self.ctx.settings,
                              executor=None, blackboard=None)
        return MasterOrchestrator(ctx)

    def test_planner_prompt_includes_lessons(self):
        analyzer = FailureAnalyzer(self.ctx)
        analyzer.learn_from_failure(
            FailureCase(source="tool", summary="plan broke",
                        error="KeyError: 'goal'"),
            analysis={"category": "missing_key", "root_cause": "no goal",
                      "lesson": "always pass goal",
                      "fix": "pass goal",
                      "prevention": "never decompose without a clear goal"})
        orch = self._orch()
        plan = orch.plan("decompose without a clear goal here")
        self.assertTrue(self.captured, "router was never asked")
        self.assertIn("never decompose without a clear goal",
                      self.captured[0])
        # and the plan still falls back cleanly
        self.assertTrue(plan.steps)


class ScheduleTestCase(unittest.TestCase):
    def test_ensure_schedule_idempotent(self):
        ctx = _ctx()
        first = ensure_improvement_schedule(ctx)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(j.get("scheduled") for j in first))
        second = ensure_improvement_schedule(ctx)
        self.assertTrue(all(j.get("already_scheduled") for j in second))
        rows = ctx.db.query(
            "SELECT name, payload FROM schedule_jobs WHERE name LIKE "
            "'improve-%'")
        self.assertEqual(len(rows), 2)
        payloads = {r["name"]: json.loads(r["payload"]) for r in rows}
        self.assertEqual(payloads["improve-evolution-tick"]["tool"],
                         "skill_evolve")
        self.assertEqual(payloads["improve-synthesis-scan"]["tool"],
                         "skill_synthesize")


class CLIImproveTestCase(unittest.TestCase):
    """Exercise `nm improve` through the real CLI entry point."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="improvecli_"))
        self.addCleanup(shutil.rmtree, self.home, True)
        (self.home / ".nomorals" / "workspace").mkdir(parents=True)
        self.env = dict(os.environ, HOME=str(self.home),
                        PYTHONPATH="/home/hatch/workspace/devon")

    def _nm(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "nomorals", "improve", *args],
            capture_output=True, text=True, timeout=180,
            cwd="/home/hatch/workspace/devon", env=self.env)

    def test_cli_status(self):
        proc = self._nm("status")
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        self.assertIn("mode:", proc.stdout)
        self.assertIn("candidates:", proc.stdout)
        self.assertIn("lessons:", proc.stdout)

    def test_cli_lessons_empty(self):
        proc = self._nm("lessons", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        self.assertEqual(json.loads(proc.stdout), [])


if __name__ == "__main__":
    unittest.main()
