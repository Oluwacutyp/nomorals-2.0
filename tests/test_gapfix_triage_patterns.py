"""Pattern-port tests: repo attention + PR landability (lane 123).

The designs are mined, not copied, from storytold's two craft apps
(both declare MIT in their manifests with no licence file — pattern
only, no upstream code or text vendored):

* craft-repo-status-app's urgency table (``craft-core/src/urgency.rs``):
  crash / data-loss words, broken words, bug labels, triage levels,
  user reports, engagement, latest build, freshness — one score per
  issue, the repo urgency the sum of the top five.
* craft-pr-review-tool's landability truth table
  (``crates/pr-core/src/assess.rs``): nine first-match rows over
  draft / problem / conflicted / inspected / running / CI / approved /
  quality, with quality, ease and focus scores around it.

The tests prove the tables' edges: every threshold boundary, every
truth-table row (all 256 signal combinations), and the tool surface
that exercises both — with the tool count pinned at 774. 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from nomorals.triage import (  # noqa: E402
    prassess,
    urgency,
)
from nomorals.triage.prassess import (  # noqa: E402
    AuthorFacts,
    PrConfig,
    PrInput,
)
from nomorals.triage.urgency import IssueInput  # noqa: E402

NOW = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)
REPO = "Oluwacutyp/nomorals-2.0"


def _issue(title: str, assoc: str = "NONE", **over: Any) -> IssueInput:
    row: dict[str, Any] = {
        "number": len(title),
        "title": title,
        "author_association": assoc,
    }
    row.update(over)
    return IssueInput.from_dict(row)


def _score(issue: IssueInput, **kw: Any) -> float:
    scored = urgency.score_issue(issue, NOW, kw.get("latest_tag"))
    assert scored is not None
    return scored.score


class TestUrgencyScoring(unittest.TestCase):
    def test_crash_report_beats_automated_docs_fix(self) -> None:
        crash = _issue("[Windows 11 0.50] Merge Down freezes on large "
                       "canvas")
        docs = _issue("fix(docs): layer-type errors say "
                      "\"a Adjustment layer\"", "CONTRIBUTOR")
        scored = urgency.score_issue(crash, NOW, "v0.5.0")
        assert scored is not None
        self.assertTrue(scored.critical)
        self.assertIn("latest build", scored.reasons)
        self.assertIn("user report", scored.reasons)
        self.assertGreater(_score(crash, latest_tag="v0.5.0"),
                           _score(docs) + 40.0)
        docs_scored = urgency.score_issue(docs, NOW, None)
        assert docs_scored is not None
        self.assertFalse(docs_scored.critical)

    def test_critical_words_worth_40_broken_words_worth_15(self) -> None:
        crash = _issue("App crashes on launch")
        self.assertEqual(_score(crash), 50.0)  # 40 crash + 10 user
        broken = _issue("delete button doesn't delete a layer")
        self.assertEqual(_score(broken), 25.0)  # 15 broken + 10 user
        plain = _issue("Sony ARW files unreadable", "CONTRIBUTOR")
        self.assertEqual(_score(plain), 0.0)

    def test_critical_categories_count_once(self) -> None:
        one = _issue("App crashes and crashes again on launch")
        self.assertEqual(_score(one), 50.0)
        two = _issue("App crashes and corrupts my files")
        scored = urgency.score_issue(two, NOW)
        assert scored is not None
        self.assertEqual(scored.score, 50.0)
        self.assertIn("crash, corruption", scored.reasons)

    def test_feature_requests_sink_and_wontfix_vanishes(self) -> None:
        bug = _issue("delete button doesn't delete a layer on windows")
        feat = _issue("Request: Gradient support for shape fill")
        self.assertGreater(_score(bug), _score(feat))
        self.assertEqual(_score(feat), 0.0)
        dup = _issue("App crashes", labels=["duplicate"])
        self.assertIsNone(urgency.score_issue(dup, NOW))
        for label in ("wontfix", "invalid", "not planned"):
            gone = _issue("App crashes", labels=[label])
            self.assertIsNone(urgency.score_issue(gone, NOW),
                              label)
        feat_crash = _issue("Add crash reporting")
        scored = urgency.score_issue(feat_crash, NOW)
        assert scored is not None
        self.assertFalse(scored.critical)

    def test_triage_labels_outrank_words_in_the_title(self) -> None:
        plain = _issue("Sony ARW files unreadable", "CONTRIBUTOR",
                       labels=[])
        urgent = _issue("Sony ARW files unreadable", "CONTRIBUTOR",
                        labels=["priority: critical"])
        plain_scored = urgency.score_issue(plain, NOW)
        urgent_scored = urgency.score_issue(urgent, NOW)
        assert plain_scored is not None
        assert urgent_scored is not None
        self.assertFalse(plain_scored.critical)
        self.assertTrue(urgent_scored.critical)
        self.assertEqual(urgent_scored.score,
                         plain_scored.score + 25.0)
        p0 = _issue("Odd", "CONTRIBUTOR", labels=["priority:P0"])
        p0_scored = urgency.score_issue(p0, NOW)
        assert p0_scored is not None
        self.assertTrue(p0_scored.critical)

        crash = _issue("Crash on export", "CONTRIBUTOR")
        low = _issue("Crash on export", "CONTRIBUTOR",
                     labels=["priority: low"])
        crash_scored = urgency.score_issue(crash, NOW)
        low_scored = urgency.score_issue(low, NOW)
        assert crash_scored is not None
        assert low_scored is not None
        self.assertTrue(crash_scored.critical)
        self.assertFalse(low_scored.critical)
        self.assertEqual(low_scored.score, crash_scored.score - 15.0)

        high = _issue("Crash on export", "CONTRIBUTOR",
                      labels=["priority: high"])
        high_scored = urgency.score_issue(high, NOW)
        assert high_scored is not None
        self.assertTrue(high_scored.critical)
        self.assertEqual(high_scored.score, crash_scored.score + 10.0)

    def test_bug_and_reliability_labels(self) -> None:
        # The label feeds both the phrase scan (labels are part of
        # the scored text) and the label bonus: "bug" is a broken
        # word (+15) and a bug label (+15).
        base = _issue("Sony ARW files unreadable", "CONTRIBUTOR")
        bug = _issue("Sony ARW files unreadable", "CONTRIBUTOR",
                     labels=["bug"])
        self.assertEqual(_score(bug), _score(base) + 30.0)
        regression = _issue("Sony ARW files unreadable", "CONTRIBUTOR",
                            labels=["regression"])
        self.assertEqual(_score(regression), _score(base) + 30.0)
        perf = _issue("Sony ARW files unreadable", "CONTRIBUTOR",
                      labels=["performance"])
        self.assertEqual(_score(perf), _score(base) + 6.0)

    def test_issues_waiting_on_the_reporter_rank_lower(self) -> None:
        crash = _issue("Crash on export", "CONTRIBUTOR")
        waiting = _issue("Crash on export", "CONTRIBUTOR",
                         labels=["Needs Info"])
        crash_scored = urgency.score_issue(crash, NOW)
        waiting_scored = urgency.score_issue(waiting, NOW)
        assert crash_scored is not None
        assert waiting_scored is not None
        self.assertEqual(waiting_scored.score,
                         crash_scored.score - 15.0)
        self.assertIn("needs info", waiting_scored.reasons)
        self.assertTrue(waiting_scored.critical)
        self.assertTrue(urgency.needs_info(["needs-info"]))
        self.assertFalse(urgency.needs_info(["bug"]))

    def test_user_report_only_for_outsiders(self) -> None:
        for assoc in ("NONE", "FIRST_TIMER", "FIRST_TIME_CONTRIBUTOR"):
            scored = urgency.score_issue(_issue("Bug here", assoc), NOW)
            assert scored is not None
            self.assertIn("user report", scored.reasons, assoc)
        for assoc in ("CONTRIBUTOR", "MEMBER", "OWNER"):
            scored = urgency.score_issue(_issue("Bug here", assoc), NOW)
            assert scored is not None
            self.assertNotIn("user report", scored.reasons, assoc)

    def test_engagement_is_capped_and_counted(self) -> None:
        quiet = _issue("Bug: layer locked but editing still works")
        loud = _issue("Bug: layer locked but editing still works",
                      reactions=15, comments=13)
        louder = _issue("Bug: layer locked but editing still works",
                        reactions=5000)
        self.assertEqual(_score(quiet), 25.0)
        self.assertGreater(_score(loud), _score(quiet) + 30.0)
        self.assertAlmostEqual(_score(loud), 65.0, places=1)
        self.assertLessEqual(_score(louder) - _score(quiet),
                             40.0 + 1e-9)
        scored = urgency.score_issue(loud, NOW)
        assert scored is not None
        self.assertIn("15 👍", scored.reasons)
        self.assertIn("13 💬", scored.reasons)

    def test_engagement_log_shape(self) -> None:
        one_reaction = _issue("Bug here", reactions=1)
        expected = 25.0 + 10.0 * math.log(4.0)
        self.assertAlmostEqual(_score(one_reaction), expected)

    def test_version_matching(self) -> None:
        self.assertEqual(urgency._major_minor("v0.5.0"), "0.5")
        self.assertEqual(urgency._major_minor("v0.1.0-rc.3"), "0.1")
        self.assertIsNone(urgency._major_minor("nightly"))
        self.assertTrue(urgency._mentions_version(
            "[windows 0.5.0] x", "0.5"))
        self.assertTrue(urgency._mentions_version(
            "[windows 11 0.50] x", "0.5"))
        self.assertFalse(urgency._mentions_version("v10.5 x", "0.5"))
        self.assertFalse(urgency._mentions_version("0.51 x", "0.5"))
        named = _issue("[windows 0.5] App crashes")
        scored = urgency.score_issue(named, NOW, "v0.5.0")
        assert scored is not None
        self.assertIn("latest build", scored.reasons)
        self.assertEqual(scored.score, 60.0)  # 40 + 10 user + 10 build

    def test_freshness_and_staleness(self) -> None:
        fresh = _issue("Bug here", "CONTRIBUTOR",
                       created_at=NOW - timedelta(hours=1))
        self.assertEqual(_score(fresh), 15.0 + 8.0)
        scored = urgency.score_issue(fresh, NOW)
        assert scored is not None
        self.assertIn("new", scored.reasons)
        two_days = _issue("Bug here", "CONTRIBUTOR",
                          created_at=NOW - timedelta(hours=48))
        self.assertEqual(_score(two_days), 15.0 + 4.0)
        old = _issue("Bug here", "CONTRIBUTOR",
                     created_at=NOW - timedelta(days=10))
        self.assertEqual(_score(old), 15.0)
        stale = _issue("Bug here", "CONTRIBUTOR",
                       created_at=NOW - timedelta(days=40),
                       updated_at=NOW - timedelta(days=31))
        self.assertEqual(_score(stale), 15.0 - 5.0)

    def test_score_never_goes_below_zero(self) -> None:
        feat = _issue("Add gradients", "CONTRIBUTOR",
                      created_at=NOW - timedelta(days=40),
                      updated_at=NOW - timedelta(days=31),
                      labels=["needs info"])
        self.assertEqual(_score(feat), 0.0)

    def test_normalize_is_word_bounded(self) -> None:
        text = urgency._normalize("App CRASHES—on start’s (Windows)")
        self.assertIn(" crashes ", text)
        self.assertIn(" start's ", text)
        hangar = urgency._normalize("hangar")
        self.assertNotIn(" hang ", hangar)


class TestUrgencyRanking(unittest.TestCase):
    def test_ranking_dedupes_sorts_and_sums_top_five(self) -> None:
        items = []
        for n in range(1, 9):
            items.append(IssueInput(
                number=n,
                title=("App crashes" if n % 2 == 0 else "Add gradients"),
                author_association="NONE"))
        items.append(items[1])
        ranked, repo_urgency, critical = urgency.rank_issues(
            items, NOW, None, 3)
        self.assertEqual(critical, 4)
        self.assertEqual(len(ranked), 3)
        self.assertTrue(all(row.critical for row in ranked))
        scores = [row.score for row in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(repo_urgency, 4.0 * 50.0)

    def test_ties_break_newest_first(self) -> None:
        old = IssueInput(number=1, title="App crashes",
                         author_association="NONE",
                         created_at=NOW - timedelta(days=10))
        new = IssueInput(number=2, title="App crashes",
                         author_association="NONE",
                         created_at=NOW - timedelta(days=9))
        ranked, _, _ = urgency.rank_issues([old, new], NOW)
        self.assertEqual([row.score for row in ranked], [50, 50])
        self.assertEqual([row.number for row in ranked], [2, 1])

    def test_zero_score_issues_drop_out(self) -> None:
        items = [IssueInput(number=1, title="Add gradients",
                            author_association="CONTRIBUTOR"),
                 IssueInput(number=2, title="App crashes",
                            author_association="NONE")]
        ranked, repo_urgency, critical = urgency.rank_issues(items, NOW)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].number, 2)
        self.assertEqual(repo_urgency, 50.0)
        self.assertEqual(critical, 1)


if __name__ == "__main__":
    unittest.main()

LONG_BODY = (
    "A thorough description of what changes here and why, with enough "
    "words in it to read as a real explanation of the change, its "
    "motivation, its alternatives, and how it was tested by hand on "
    "all platforms."
)

CODE_FILE = {"path": "crates/a/src/lib.rs", "additions": 40,
             "deletions": 0}
TEST_FILE = {"path": "crates/a/tests/lib.rs", "additions": 20,
             "deletions": 0}


def _pr(**over: Any) -> PrInput:
    row: dict[str, Any] = {
        "number": 1,
        "author": "ann",
        "association": "CONTRIBUTOR",
        "mergeable": "MERGEABLE",
        "merge_state": "CLEAN",
        "head_oid": "h",
        "additions": 60,
        "deletions": 5,
        "changed_files": 2,
        "ci_rollup": "SUCCESS",
        "updated_at": "2026-10-30T00:00:00Z",
        "listed_at": "2026-10-30T01:00:00Z",
    }
    row.update(over)
    return PrInput.from_dict(row)


def _detail(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fetched_at": "2026-10-30T01:00:00Z",
        "for_updated_at": "2026-10-30T00:00:00Z",
        "for_head": "h",
        "body": LONG_BODY,
        "files": [dict(CODE_FILE), dict(TEST_FILE)],
        "checks": [{"name": "test", "state": "SUCCESS"}],
        "rabbit": "reviewed",
        "findings": [],
        "human_threads_open": 0,
        "linked_issues": [],
    }
    row.update(over)
    return row


def _clean() -> PrInput:
    return _pr(detail=_detail())


def _approved(pr: PrInput) -> PrInput:
    pr.review_decision = "APPROVED"
    pr.approved_on_head = ["bo"]
    return pr


def _assess(pr: PrInput, author: AuthorFacts | None = None,
            config: PrConfig | None = None) -> dict[str, Any]:
    return prassess.assess_pr(pr, author or AuthorFacts(),
                              config or PrConfig())


class TestPrTruthTable(unittest.TestCase):
    def test_truth_table_covers_all_256_combinations(self) -> None:
        y, n, x = True, False, None
        # draft, problem, conflicted, inspected, running, ci passed,
        # approved, high quality — None means the column does not
        # matter and is tried both ways.
        table = [
            ((y, x, x, x, x, x, x, x), "draft"),
            ((n, y, x, x, x, x, x, x), "needs_work"),
            ((n, n, y, x, x, x, x, x), "conflicted"),
            ((n, n, n, n, x, x, x, x), "not_inspected"),
            ((n, n, n, y, y, x, x, x), "checking"),
            ((n, n, n, y, n, n, x, x), "ci_not_run"),
            ((n, n, n, y, n, y, y, x), "landable"),
            ((n, n, n, y, n, y, n, y), "promising"),
            ((n, n, n, y, n, y, n, n), "reviewable"),
        ]
        fields = ("draft", "problem", "conflicted", "inspected",
                  "running", "ci_passed", "approved", "high_quality")
        seen = 0
        for bits in itertools.product((False, True), repeat=8):
            row = dict(zip(fields, bits, strict=True))
            signals = prassess.Signals.from_dict(row)
            matches = [state for cols, state in table
                       if all(c is None or c == v
                              for c, v in zip(cols, bits, strict=True))]
            self.assertEqual(len(matches), 1,
                             f"every combination matches exactly one "
                             f"row: {row}")
            self.assertEqual(prassess.headline(signals), matches[0],
                             str(row))
            seen += 1
        self.assertEqual(seen, 256)

    def test_clean_pr_with_no_approval_is_promising_not_landable(
            self) -> None:
        out = _assess(_clean())
        self.assertEqual(out["state"], "promising")
        self.assertEqual(out["ease"], "easy")
        self.assertGreaterEqual(out["quality"], 75)
        self.assertTrue(out["high_quality"])
        self.assertTrue(out["tests_touched"])

    def test_landable_takes_a_current_approval(self) -> None:
        out = _assess(_approved(_clean()))
        self.assertEqual(out["state"], "landable")
        self.assertTrue(out["approved"])
        self.assertFalse(out["approval_stale"])

    def test_a_person_outranks_the_score(self) -> None:
        pr = _approved(_clean())
        assert pr.detail is not None
        pr.detail.body = ""
        pr.detail.files = [dict(CODE_FILE)]
        out = _assess(pr)
        self.assertLess(out["quality"], 75)
        self.assertEqual(out["state"], "landable")

    def test_a_low_score_with_no_approval_is_reviewable(self) -> None:
        pr = _clean()
        assert pr.detail is not None
        pr.detail.body = ""
        pr.detail.files = [dict(CODE_FILE)]
        self.assertEqual(_assess(pr)["state"], "reviewable")

    def test_an_approval_of_an_older_commit_doesnt_count(self) -> None:
        pr = _approved(_clean())
        pr.approved_on_head = []
        out = _assess(pr)
        self.assertTrue(out["approved"])
        self.assertTrue(out["approval_stale"])
        self.assertEqual(out["state"], "promising")
        self.assertTrue(any("predates" in r
                            for r in out["state_reasons"]))

    def test_problem_sources_each_block_landing(self) -> None:
        ci_fail = _clean()
        assert ci_fail.detail is not None
        ci_fail.detail.checks = [{"name": "test", "state": "FAILURE"}]
        self.assertEqual(_assess(ci_fail)["state"], "needs_work")
        self.assertIn("CI is failing",
                      _assess(ci_fail)["state_reasons"])

        requested = _clean()
        requested.review_decision = "CHANGES_REQUESTED"
        out = _assess(requested)
        self.assertEqual(out["state"], "needs_work")
        self.assertIn("A reviewer requested changes",
                      out["state_reasons"])

        flagged = _clean()
        assert flagged.detail is not None
        flagged.detail.findings = [{"severity": "major",
                                    "security": False}]
        out = _assess(flagged)
        self.assertEqual(out["state"], "needs_work")
        self.assertIn("1 unresolved CodeRabbit flag(s)",
                      out["state_reasons"])

        sec_fail = _clean()
        assert sec_fail.detail is not None
        sec_fail.detail.checks = [
            {"name": "CodeQL", "state": "FAILURE"}]
        out = _assess(sec_fail)
        self.assertEqual(out["state"], "needs_work")
        self.assertTrue(any("Security check failed" in r
                            for r in out["state_reasons"]))

        talking = _clean()
        assert talking.detail is not None
        talking.detail.human_threads_open = 2
        out = _assess(talking)
        self.assertEqual(out["state"], "needs_work")
        self.assertIn("2 unresolved review conversation(s)",
                      out["state_reasons"])

    def test_running_sources_each_mean_checking(self) -> None:
        pending = _clean()
        assert pending.detail is not None
        pending.detail.checks = [{"name": "test", "state": "PENDING"}]
        self.assertEqual(_assess(pending)["state"], "checking")

        rabbit = _clean()
        assert rabbit.detail is not None
        rabbit.detail.rabbit = "inprogress"
        self.assertEqual(_assess(rabbit)["state"], "checking")

        unknown = _clean()
        unknown.mergeable = "UNKNOWN"
        self.assertEqual(_assess(unknown)["state"], "checking")

    def test_conflicted_from_either_merge_signal(self) -> None:
        for over in ({"mergeable": "CONFLICTING"},
                     {"merge_state": "DIRTY"}):
            pr = _clean()
            for key, value in over.items():
                setattr(pr, key, value)
            self.assertEqual(_assess(pr)["state"], "conflicted")

    def test_not_inspected_without_current_detail(self) -> None:
        bare = _pr()
        self.assertEqual(_assess(bare)["state"], "not_inspected")
        stale = _clean()
        assert stale.detail is not None
        stale.detail.for_head = "old"
        self.assertEqual(_assess(stale)["state"], "not_inspected")

    def test_ci_not_run_when_inspected_but_no_ci(self) -> None:
        pr = _clean()
        assert pr.detail is not None
        pr.detail.checks = []
        pr.ci_rollup = None
        self.assertEqual(_assess(pr)["state"], "ci_not_run")

    def test_ci_state_ignores_rabbit_cancelled_and_skipped(self) -> None:
        pr = _clean()
        assert pr.detail is not None
        pr.detail.checks = [
            {"name": "CodeRabbit", "state": "FAILURE"},
            {"name": "test", "state": "CANCELLED"},
            {"name": "lint", "state": "SKIPPED"},
        ]
        self.assertEqual(prassess.ci_state(pr), "none")

    def test_ci_state_falls_back_to_the_rollup(self) -> None:
        pr = _clean()
        assert pr.detail is not None
        pr.detail.fetched_at = "2026-10-29T00:00:00Z"
        for rollup, want in (("SUCCESS", "success"),
                             ("FAILURE", "failure"),
                             ("ERROR", "failure"),
                             ("ANYTHING_ELSE", "pending"),
                             (None, "none")):
            pr.ci_rollup = rollup
            self.assertEqual(prassess.ci_state(pr), want, rollup)


class TestPrQualityEase(unittest.TestCase):
    def test_high_quality_threshold_is_75_exactly(self) -> None:
        base = _detail(rabbit="none", files=[{
            "path": "README.md", "additions": 60, "deletions": 0}])
        at = _pr(detail=_detail(
            **{**base, "findings": [{"severity": "minor"}]}))
        out = _assess(at)
        self.assertEqual(out["quality"], 75)
        self.assertTrue(out["high_quality"])
        self.assertEqual(out["state"], "promising")
        below = _pr(detail=_detail(
            **{**base, "findings": [{"severity": "minor"},
                                    {"severity": "trivial"}]}))
        out = _assess(below)
        self.assertEqual(out["quality"], 74)
        self.assertFalse(out["high_quality"])
        self.assertEqual(out["state"], "reviewable")

    def test_quality_arithmetic(self) -> None:
        out = _assess(_clean())
        self.assertEqual(out["quality"], 98)  # 60 + 10 + 8 + 12 + 8
        by_reason = {r["text"]: r["points"]
                     for r in out["quality_reasons"]}
        self.assertEqual(by_reason["CI passes"], 10)
        self.assertEqual(by_reason["Explains itself"], 8)
        self.assertEqual(by_reason["Includes tests"], 12)
        self.assertEqual(by_reason["CodeRabbit had nothing to add"], 8)

    def test_finding_penalties_are_capped(self) -> None:
        pr = _clean()
        assert pr.detail is not None
        pr.detail.findings = [{"severity": "critical"}] * 5
        out = _assess(pr)
        penalties = {r["text"]: r["points"]
                     for r in out["quality_reasons"]}
        self.assertEqual(penalties["5 critical finding(s)"], -40)
        pr.detail.findings = [{"severity": "major"}] * 5
        out = _assess(pr)
        penalties = {r["text"]: r["points"]
                     for r in out["quality_reasons"]}
        self.assertEqual(penalties["5 major finding(s)"], -30)

    def test_ease_line_bands(self) -> None:
        cases = [(49, 0, "easy"), (50, 1, "easy"), (199, 1, "easy"),
                 (200, 2, "easy"), (599, 2, "easy"), (600, 4, "medium"),
                 (1999, 4, "medium"), (2000, 6, "hard")]
        for lines, points, label in cases:
            pr = _pr(additions=lines, deletions=0, changed_files=1,
                     detail=_detail(files=[]))
            out = _assess(pr)
            self.assertEqual(out["ease_score"], points, lines)
            self.assertEqual(out["ease"], label, lines)

    def test_ease_file_bands_and_spread(self) -> None:
        for files, points in ((3, 0), (4, 1), (10, 1), (11, 2),
                              (30, 2), (31, 4)):
            pr = _pr(additions=0, deletions=0, changed_files=files,
                     detail=_detail(files=[]))
            self.assertEqual(_assess(pr)["ease_score"], points, files)
        spread = _pr(additions=0, deletions=0, changed_files=1,
                     detail=_detail(files=[
                         {"path": f"area{i}/x/lib.rs",
                          "additions": 0, "deletions": 0}
                         for i in range(5)]))
        out = _assess(spread)
        self.assertEqual(out["ease_score"], 1)
        self.assertIn("Spreads across 5 areas",
                      [r["text"] for r in out["ease_reasons"]])

    def test_ease_extras(self) -> None:
        conflicted = _clean()
        conflicted.mergeable = "CONFLICTING"
        self.assertEqual(_assess(conflicted)["ease_score"], 3)
        flagged = _clean()
        assert flagged.detail is not None
        flagged.detail.findings = [{"severity": "major"}] * 5
        self.assertEqual(_assess(flagged)["ease_score"], 4)
        heavy = _clean()
        assert heavy.detail is not None
        heavy.detail.findings = [{"severity": "minor",
                                  "effort": "heavy lift"}]
        self.assertEqual(_assess(heavy)["ease_score"], 2)

    def test_author_tiers(self) -> None:
        bot = _pr(author_is_bot=True)
        self.assertEqual(
            prassess.author_tier(bot, AuthorFacts(), PrConfig()),
            "bot")
        named_bot = _pr(author="someone[bot]")
        self.assertEqual(
            prassess.author_tier(named_bot, AuthorFacts(),
                                 PrConfig()),
            "bot")
        core = _pr(association="MEMBER")
        self.assertEqual(
            prassess.author_tier(core, AuthorFacts(), PrConfig()),
            "core")
        writer = _pr()
        self.assertEqual(
            prassess.author_tier(
                writer, AuthorFacts(permission="WRITE"), PrConfig()),
            "core")
        listed = _pr()
        self.assertEqual(
            prassess.author_tier(
                listed, AuthorFacts(listed=True), PrConfig()),
            "core")
        strict = _pr()
        self.assertEqual(
            prassess.author_tier(
                strict, AuthorFacts(listed=True),
                PrConfig(people_toml_is_core=False)),
            "regular")
        frequent = _pr()
        self.assertEqual(
            prassess.author_tier(
                frequent, AuthorFacts(landed_total=31), PrConfig()),
            "frequent")
        self.assertEqual(
            prassess.author_tier(
                frequent, AuthorFacts(landed_total=30), PrConfig()),
            "regular")
        new = _pr(association="FIRST_TIME_CONTRIBUTOR")
        self.assertEqual(
            prassess.author_tier(new, AuthorFacts(), PrConfig()),
            "new")

    def test_sensitive_paths_flag_review_not_penalty(self) -> None:
        path = {"path": ".github/workflows/ci.yml",
                "additions": 5, "deletions": 0}
        plain = _clean()
        assert plain.detail is not None
        plain.detail.files = [dict(CODE_FILE)]
        sensitive = _clean()
        assert sensitive.detail is not None
        sensitive.detail.files = [dict(CODE_FILE), dict(path)]
        out_plain = _assess(plain)
        out = _assess(sensitive)
        self.assertEqual(out["sensitive_paths"],
                         [".github/workflows/ci.yml"])
        self.assertEqual(out["quality"], out_plain["quality"])
        self.assertEqual(out["ease_score"],
                         out_plain["ease_score"] + 2)
        writer = _clean()
        assert writer.detail is not None
        writer.detail.files = [dict(CODE_FILE), dict(path)]
        out_writer = _assess(
            writer, AuthorFacts(permission="WRITE"))
        self.assertEqual(out_writer["sensitive_paths"], [])

    def test_focus_orders_the_review_queue(self) -> None:
        states = {
            "draft": _pr(is_draft=True),
            "needs_work": _clean(),
            "conflicted": _pr(mergeable="CONFLICTING"),
            "not_inspected": _pr(),
            "checking": _pr(detail=_detail(
                checks=[{"name": "test", "state": "PENDING"}])),
            "ci_not_run": _pr(detail=_detail(checks=[]),
                              ci_rollup=None),
            "landable": _approved(_clean()),
            "promising": _clean(),
            "reviewable": _pr(detail=_detail(
                body="", files=[dict(CODE_FILE)])),
        }
        states["needs_work"].detail = prassess.PrDetail.from_dict(
            _detail(checks=[{"name": "test", "state": "FAILURE"}]))
        focus = {name: _assess(pr)["focus"]
                 for name, pr in states.items()}
        order = ["landable", "promising", "reviewable", "ci_not_run",
                 "checking", "not_inspected", "conflicted",
                 "needs_work", "draft"]
        values = [focus[name] for name in order]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(_assess(_approved(_clean()))["focus"], 8210)

    def test_interesting_from_reactions_or_comments(self) -> None:
        self.assertTrue(_assess(_pr(reactions=3))["interesting"])
        self.assertTrue(_assess(_pr(comments_total=8))["interesting"])
        self.assertFalse(_assess(_pr())["interesting"])

    def test_outside_review_can_only_make_it_harder(self) -> None:
        pr = _clean()
        pr.analysis = {"ease": "hard", "interesting": True}
        out = _assess(pr)
        self.assertEqual(out["ease"], "hard")
        self.assertTrue(out["ease_raised_by_review"])
        self.assertEqual(out["state"], "promising")
        self.assertEqual(out["quality"], 98)
        self.assertTrue(out["interesting"])
        huge = _pr(additions=5000, deletions=0, changed_files=1,
                   detail=_detail(files=[]))
        huge.analysis = {"ease": "easy"}
        out = _assess(huge)
        self.assertEqual(out["ease"], "hard")
        self.assertFalse(out["ease_raised_by_review"])


class _FakeGitHub:
    """A dict-backed stand-in for the connector's issue surface."""

    def __init__(self) -> None:
        self.issues: dict[tuple[str, int], dict[str, Any]] = {}

    def seed(self, number: int, title: str, **over: Any) -> None:
        row: dict[str, Any] = {
            "number": number,
            "title": title,
            "body": "",
            "state": "open",
            "labels": [],
            "author": "someone",
            "author_association": "NONE",
            "reactions": 0,
            "comments": 0,
            "created_at": "2026-11-01T00:00:00Z",
            "updated_at": "2026-11-18T00:00:00Z",
            "url": f"https://github.com/{REPO}/issues/{number}",
            "is_pull_request": False,
        }
        row.update(over)
        self.issues[(REPO, number)] = row

    def list_issues(self, full_name: str, state: str = "open",
                    limit: int = 50) -> list[dict[str, Any]]:
        return [dict(row) for (repo, _), row in self.issues.items()
                if repo == full_name and row["state"] == state]


class TestToolSurface(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        from nomorals.triage.store import TriageStore
        # The store path the tool builds from settings: <home>/triage.
        self.store = TriageStore(
            Path(self._tmp.name) / "triage" / "triage.json")
        self.github = _FakeGitHub()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _registry(self):
        from nomorals.tools import registry as tool_registry
        from nomorals.tools import triage as triage_tool
        reg = tool_registry.ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(
                home_path=Path(self._tmp.name)),
            _triage_github=self.github,
            _triage_suggest=None,
            db=None, router=None))
        triage_tool.register(reg)
        return reg

    def _call(self, reg, **kw: Any) -> dict[str, Any]:
        outcome = reg.call("triage", **kw)
        return outcome.value if hasattr(outcome, "value") else outcome

    def _seed_urgent_rows(self) -> None:
        self.store.track(REPO)
        from nomorals.triage.engine import TriageEngine
        engine = TriageEngine(self.github, self.store, suggest=None)
        # Relative dates, so the scores hold whenever the suite runs:
        # ten days old (no freshness) and updated yesterday (not stale).
        now = datetime.now(UTC)
        created = (now - timedelta(days=10)).isoformat().replace(
            "+00:00", "Z")
        updated = (now - timedelta(days=1)).isoformat().replace(
            "+00:00", "Z")
        self.github.seed(1, "App crashes on launch",
                         created_at=created, updated_at=updated)
        self.github.seed(2, "Add gradients",
                         created_at=created, updated_at=updated)
        self.github.seed(3, "Bug here", author_association="MEMBER",
                         created_at=created, updated_at=updated)
        engine.sync(REPO)

    def test_urgent_ranks_the_mirrored_repo(self) -> None:
        self._seed_urgent_rows()
        out = self._call(self._registry(), action="urgent", repo=REPO)
        self.assertTrue(out["ok"])
        self.assertEqual(out["repo"], REPO)
        self.assertEqual([row["number"] for row in out["ranked"]],
                         [1, 3])
        self.assertEqual(out["ranked"][0]["score"], 50)
        self.assertEqual(out["ranked"][0]["reasons"],
                         ["crash", "user report"])
        self.assertEqual(out["urgency"], 65.0)
        self.assertEqual(out["critical"], 1)
        self.assertEqual(out["scored"], 2)

    def test_urgency_scores_one_issue_ad_hoc(self) -> None:
        out = self._call(self._registry(), action="urgency",
                         title="App crashes on launch",
                         author_association="NONE")
        self.assertTrue(out["ok"])
        self.assertEqual(out["score"], 50.0)
        self.assertTrue(out["critical"])
        gone = self._call(self._registry(), action="urgency",
                          title="App crashes", labels="duplicate")
        self.assertTrue(gone["ok"])
        self.assertIsNone(gone["score"])
        bad = self._call(self._registry(), action="urgency",
                         title="Bug", comments="many")
        self.assertFalse(bad["ok"])

    def test_prstate_runs_the_truth_table(self) -> None:
        reg = self._registry()
        cases = [
            ({"draft": "1"}, "draft"),
            ({"problem": "1"}, "needs_work"),
            ({"conflicted": "1"}, "conflicted"),
            ({}, "not_inspected"),
            ({"inspected": "1", "running": "1"}, "checking"),
            ({"inspected": "1"}, "ci_not_run"),
            ({"inspected": "1", "ci_passed": "1",
              "approved": "1"}, "landable"),
            ({"inspected": "1", "ci_passed": "1",
              "high_quality": "1"}, "promising"),
            ({"inspected": "1", "ci_passed": "1"}, "reviewable"),
        ]
        for flags, want in cases:
            out = self._call(reg, action="prstate", **flags)
            self.assertTrue(out["ok"], flags)
            self.assertEqual(out["state"], want, flags)

    def test_pr_grades_facts_as_json(self) -> None:
        reg = self._registry()
        pr_row = {"number": 7, "author": "ann",
                  "association": "CONTRIBUTOR",
                  "mergeable": "MERGEABLE", "merge_state": "CLEAN",
                  "head_oid": "h", "additions": 60, "deletions": 5,
                  "changed_files": 2, "ci_rollup": "SUCCESS",
                  "updated_at": "2026-10-30T00:00:00Z",
                  "listed_at": "2026-10-30T01:00:00Z",
                  "review_decision": "APPROVED",
                  "approved_on_head": ["bo"],
                  "detail": _detail()}
        out = self._call(reg, action="pr", pr=json.dumps(pr_row))
        self.assertTrue(out["ok"])
        self.assertEqual(out["state"], "landable")
        self.assertEqual(out["focus"], 8210)
        bad = self._call(reg, action="pr", pr="{not json")
        self.assertFalse(bad["ok"])
        unknown = self._call(reg, action="nonsense")
        self.assertFalse(unknown["ok"])
        self.assertIn("urgent", unknown["error"])
        self.assertIn("prstate", unknown["error"])

    def test_tool_count_stays_800(self) -> None:
        from nomorals.tools.registry import ToolRegistry
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # Lane 123 adds no tool: the four new actions ride the
        # existing action-dispatched "triage" tool. 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after B.19 added scene_spawn; 776 after B.20 added cloudcam; 777 after B.22 added show; 778 after B.26 added scene_project; 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet).
        self.assertEqual(len(registry._tools), 800)
        self.assertIn("triage", registry._tools)

    def test_sync_stores_reactions_and_association(self) -> None:
        from nomorals.triage.engine import TriageEngine
        self.store.track(REPO)
        self.github.seed(1, "App crashes on launch", reactions=9,
                         author_association="NONE")
        engine = TriageEngine(self.github, self.store, suggest=None)
        out = engine.sync(REPO)
        self.assertEqual(out["new"], 1)
        row = self.store.get_item(f"{REPO}#1")
        assert row is not None
        self.assertEqual(row["reactions"], 9)
        self.assertEqual(row["author_association"], "NONE")
        self.github.seed(1, "App crashes on launch", reactions=12,
                         author_association="NONE")
        out = engine.sync(REPO)
        self.assertEqual(out["unchanged"], 1)
        row = self.store.get_item(f"{REPO}#1")
        assert row is not None
        self.assertEqual(row["reactions"], 12)

    def test_verb_urgent(self) -> None:
        self._seed_urgent_rows()
        from nomorals.agents.partner.runtime_system import (
            RuntimeSystemMixin,
        )
        runtime = object.__new__(RuntimeSystemMixin)
        runtime._triage_store = lambda: self.store
        runtime._triage_github_client = self.github
        runtime._triage_suggest = lambda: None
        text = runtime._control_triage("urgent",
                                       chat_key="telegram:1")
        self.assertIn("Most urgent", text)
        self.assertIn("#1 [50]", text)
        self.assertIn("crash, user report", text)


if __name__ == "__main__":
    unittest.main()
