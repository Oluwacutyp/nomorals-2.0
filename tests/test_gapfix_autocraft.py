"""Failing-first tests for the autocraft port (lane 83).

Autocraft (storytold/autocraft, dual MIT/Apache-2.0) is a model harness
that mirrors a GitHub backlog into a local database and runs a pipeline
over it: Sync, Duplicates, Classify, Priority, Decide, Label, Review,
Fix, Fix-up, Verify, Merge queue, Act, Post-merge review. The portable
core — the part Devon does not already have — is the triage half:
mirror the issues of tracked repos, find duplicate groups, classify
type + flags, assign P0–P4 priority, rank the work, and ask the owner
the scope questions. The rest is NOT ported: Review/Fix/Verify/Merge
need the Claude CLI + bubblewrap + Podman sandbox workers, and Devon
has her own code-fix machinery; the dashboard is not ported (Devon
answers in chat); Claude accounts are not ported (Devon has her own
brain ladder); the Jev CLI is not ported — Devon's own
``agents/jev.py`` is an offline *task* classifier, a different thing
sharing a name, so the cheap first pass here is token similarity +
keyword hints and the second opinion is the brain.

Contract points carried verbatim from docs/pipeline.md: duplicate
candidates at similarity >= 0.75, at most 3 per item, same kind;
type labels (bugfix, performance, missing feature, small extra,
plugin candidate, infra) + security / needs-decision flags; settle
at 0.9 confidence or model agreement, scope disagreements go to
Decide; P0–P4 table with the security flag raising one level; reach
0–99; work rank ``(4 - level) * 10000 + type * 1000 + size * 100 +
reach``; duplicate decisions ask "Keep #A and close #B, #C as its
duplicates?" with the exact comment texts stored at ask time.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from nomorals.triage import classify as triage_classify  # noqa: E402
from nomorals.triage import similarity as triage_sim  # noqa: E402
from nomorals.triage.store import TriageStore, item_key, text_hash  # noqa: E402

REPO = "Oluwacutyp/nomorals-2.0"


def _item(number: int, title: str, body: str = "", **over: Any) -> dict:
    item = {
        "repo": REPO,
        "number": number,
        "kind": "issue",
        "title": title,
        "body": body,
        "state": "open",
        "labels": [],
        "author": "someone",
        "comments": 0,
        "created_at": "2026-11-01T00:00:00Z",
        "updated_at": "2026-11-18T00:00:00Z",
        "url": f"https://github.com/{REPO}/issues/{number}",
    }
    item.update(over)
    item["text_hash"] = text_hash(title, body)
    return item


class FakeGitHub:
    """A dict-backed stand-in for GitHubConnector's issue surface."""

    def __init__(self) -> None:
        self.issues: dict[tuple[str, int], dict[str, Any]] = {}
        self.comments: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.next_comment_id = 100
        self.fail_list: str = ""
        self.label_calls: list[tuple[str, int, list[str]]] = []

    def seed(self, number: int, title: str, body: str = "",
             **over: Any) -> dict:
        issue = _item(number, title, body, **over)
        self.issues[(REPO, number)] = issue
        self.comments[(REPO, number)] = []
        return dict(issue)

    def list_issues(self, full_name: str, *, state: str = "open",
                    labels: str = "", limit: int = 50) -> list[dict]:
        if self.fail_list:
            raise RuntimeError(self.fail_list)
        out = [dict(i) for (r, _), i in sorted(self.issues.items())
               if r == full_name and i["state"] == state]
        return out[:limit]

    def get_issue(self, full_name: str, number: int) -> dict:
        issue = self.issues.get((full_name, number))
        if issue is None:
            raise RuntimeError(f"issue {full_name}#{number} not found")
        out = dict(issue)
        out["comments"] = len(self.comments.get((full_name, number), []))
        return out

    def comment_issue(self, full_name: str, number: int,
                      body: str) -> dict:
        if (full_name, number) not in self.issues:
            raise RuntimeError(f"issue {full_name}#{number} not found")
        comment = {"id": self.next_comment_id, "body": body,
                   "user": {"login": "devon"},
                   "created_at": "2026-11-18T00:00:00Z"}
        self.next_comment_id += 1
        self.comments[(full_name, number)].append(comment)
        return dict(comment)

    def close_issue(self, full_name: str, number: int) -> dict:
        issue = self.issues.get((full_name, number))
        if issue is None:
            raise RuntimeError(f"issue {full_name}#{number} not found")
        issue["state"] = "closed"
        return dict(issue)

    def set_issue_labels(self, full_name: str, number: int,
                         labels: list[str]) -> dict:
        issue = self.issues.get((full_name, number))
        if issue is None:
            raise RuntimeError(f"issue {full_name}#{number} not found")
        self.label_calls.append((full_name, number, list(labels)))
        issue["labels"] = list(labels)
        return dict(issue)


class FakeSuggest:
    """A scripted model: prompt-substring -> JSON text (or garbage)."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        lowered = prompt.lower()
        for needle, text in self.mapping.items():
            if needle.lower() in lowered:
                return text
        return ""


class TestSimilarity(unittest.TestCase):
    def test_identical_text_is_one(self) -> None:
        self.assertAlmostEqual(
            triage_sim.cosine_similarity("crash on login", "crash on login"),
            1.0)

    def test_unrelated_text_is_near_zero(self) -> None:
        self.assertLess(
            triage_sim.cosine_similarity(
                "crash on login with valid password",
                "add dark mode toggle to settings page"),
            0.3)

    def test_near_duplicate_scores_high(self) -> None:
        self.assertGreaterEqual(
            triage_sim.cosine_similarity(
                "app crashes on login with valid password",
                "app crashes when logging in with a valid password"),
            0.75)

    def test_candidates_threshold_limit_and_kind(self) -> None:
        now = datetime(2026, 11, 18, tzinfo=UTC)
        base = _item(1, "app crashes on login with valid password")
        near = _item(2, "app crashes when logging in with valid password")
        far = _item(3, "add dark mode toggle to settings page")
        pr = _item(4, "app crashes on login with valid password",
                   kind="pr")
        other_repo = _item(5, "app crashes on login with valid password",
                           repo="someone/else")
        got = triage_sim.duplicate_candidates(
            [near, far, pr, other_repo], base, now=now)
        self.assertEqual([row["number"] for row, _ in got], [2])
        self.assertGreaterEqual(got[0][1], 0.75)

    def test_candidates_cap_at_three(self) -> None:
        now = datetime(2026, 11, 18, tzinfo=UTC)
        base = _item(1, "login crash password valid")
        pool = [_item(n, "login crash password valid extra words here")
                for n in range(2, 8)]
        got = triage_sim.duplicate_candidates(pool, base, now=now)
        self.assertEqual(len(got), 3)

    def test_candidates_exclude_ruled_out_pairs(self) -> None:
        now = datetime(2026, 11, 18, tzinfo=UTC)
        base = _item(1, "app crashes on login with valid password")
        near = _item(2, "app crashes when logging in with valid password")
        ruled = frozenset({(item_key(REPO, 1), item_key(REPO, 2))})
        got = triage_sim.duplicate_candidates(
            [near], base, now=now, ruled_out=ruled)
        self.assertEqual(got, [])

    def test_candidates_include_recently_closed_only(self) -> None:
        now = datetime(2026, 11, 18, tzinfo=UTC)
        base = _item(1, "app crashes on login with valid password")
        recent = _item(2, "app crashes when logging in with valid password",
                       state="closed",
                       updated_at="2026-06-01T00:00:00Z")
        old = _item(3, "app crashes when logging in with valid password",
                    state="closed",
                    updated_at="2025-01-01T00:00:00Z")
        got = triage_sim.duplicate_candidates(
            [recent, old], base, now=now)
        self.assertEqual([row["number"] for row, _ in got], [2])


class TestClassifyPure(unittest.TestCase):
    def test_type_labels_are_the_upstream_six(self) -> None:
        self.assertEqual(triage_classify.TYPE_LABELS, (
            "bugfix", "performance", "missing feature", "small extra",
            "plugin candidate", "infra"))

    def test_keyword_hint_crash_is_bugfix(self) -> None:
        hint = triage_classify.keyword_hint(
            "App crashes on startup", "segfault immediately, data lost")
        self.assertEqual(hint["type"], "bugfix")
        self.assertGreaterEqual(hint["confidence"], 0.9)

    def test_keyword_hint_security_flag(self) -> None:
        hint = triage_classify.keyword_hint(
            "SQL injection in search", "unsanitized input reaches the query")
        self.assertTrue(hint["security"])
        self.assertGreaterEqual(hint["security_confidence"], 0.9)

    def test_keyword_hint_docs_typo_is_small_extra(self) -> None:
        hint = triage_classify.keyword_hint(
            "Typo in README", "fix a spelling mistake in the docs")
        self.assertEqual(hint["type"], "small extra")

    def test_keyword_hint_feature_request(self) -> None:
        hint = triage_classify.keyword_hint(
            "Add support for dark mode", "please add a dark theme option")
        self.assertEqual(hint["type"], "missing feature")

    def test_keyword_hint_vague_is_unsure(self) -> None:
        hint = triage_classify.keyword_hint("Something feels off", "idk")
        self.assertIsNone(hint["type"])
        self.assertLess(hint["confidence"], 0.9)

    def test_settle_type_offline_confident_settles(self) -> None:
        out = triage_classify.settle_type(
            {"type": "bugfix", "confidence": 0.95}, None)
        self.assertEqual((out["type"], out["source"]), ("bugfix", "offline"))

    def test_settle_type_model_agreement_settles(self) -> None:
        out = triage_classify.settle_type(
            {"type": "bugfix", "confidence": 0.6},
            {"type": "bugfix", "confidence": 0.8})
        self.assertEqual((out["type"], out["source"]), ("bugfix", "model"))

    def test_settle_type_scope_disagreement_needs_decision(self) -> None:
        out = triage_classify.settle_type(
            {"type": "missing feature", "confidence": 0.6},
            {"type": "small extra", "confidence": 0.8})
        self.assertIsNone(out["type"])
        self.assertTrue(out["needs_decision"])

    def test_settle_type_referee_settles_other_disagreement(self) -> None:
        out = triage_classify.settle_type(
            {"type": "bugfix", "confidence": 0.6},
            {"type": "infra", "confidence": 0.7},
            referee={"type": "infra", "confidence": 0.95, "sure": True})
        self.assertEqual((out["type"], out["source"]), ("infra", "referee"))

    def test_settle_type_unsure_referee_needs_decision(self) -> None:
        out = triage_classify.settle_type(
            {"type": "bugfix", "confidence": 0.6},
            {"type": "infra", "confidence": 0.7},
            referee={"type": "infra", "confidence": 0.4, "sure": False})
        self.assertIsNone(out["type"])
        self.assertTrue(out["needs_decision"])

    def test_settle_flag_thresholds(self) -> None:
        # >= 0.9 sets it from the offline value.
        self.assertTrue(triage_classify.settle_flag(
            True, 0.95, None))
        # < 0.3 clears it.
        self.assertFalse(triage_classify.settle_flag(
            True, 0.2, True))
        # In between, the model counts.
        self.assertTrue(triage_classify.settle_flag(
            False, 0.5, True))
        self.assertFalse(triage_classify.settle_flag(
            True, 0.5, False))
        # In between with no model: unsettled.
        self.assertIsNone(triage_classify.settle_flag(
            True, 0.5, None))

    def test_priority_table_and_security_raise(self) -> None:
        self.assertEqual(
            triage_classify.priority_for(
                "bugfix", signals={"crash": True}), "P0")
        self.assertEqual(
            triage_classify.priority_for(
                "bugfix", signals={"regression": True}), "P1")
        self.assertEqual(
            triage_classify.priority_for("bugfix"), "P2")
        self.assertEqual(
            triage_classify.priority_for(
                "small extra", signals={"cosmetic": True}), "P3")
        self.assertEqual(
            triage_classify.priority_for("plugin candidate"), "P4")
        self.assertEqual(
            triage_classify.priority_for("missing feature", security=True),
            "P1")

    def test_reach_is_bounded(self) -> None:
        item = _item(1, "x", comments=500)
        self.assertLessEqual(
            triage_classify.reach(item, duplicate_count=10), 99)
        self.assertGreaterEqual(
            triage_classify.reach(_item(2, "x")), 0)

    def test_work_rank_orders_level_then_type_then_size_then_reach(self) -> None:
        rank = triage_classify.work_rank
        self.assertGreater(rank("P0", "bugfix", "S", 10),
                           rank("P1", "bugfix", "S", 99))
        self.assertGreater(rank("P2", "bugfix", "S", 0),
                           rank("P2", "missing feature", "S", 99))
        self.assertGreater(rank("P2", "bugfix", "S", 0),
                           rank("P2", "bugfix", "L", 99))
        self.assertGreater(rank("P2", "bugfix", "M", 50),
                           rank("P2", "bugfix", "M", 10))
        # An unsettled type sorts last inside its level.
        self.assertGreater(rank("P2", "bugfix", "M", 0),
                           rank("P2", None, "S", 99))


class TestStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "triage.json"
        self.store = TriageStore(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_trip_and_atomic_save(self) -> None:
        self.store.track(REPO)
        self.store.upsert_item(_item(1, "crash on login"))
        self.store.save()
        self.assertFalse(Path(str(self.path) + ".tmp").exists())
        again = TriageStore(self.path)
        self.assertIn(REPO, again.tracked())
        self.assertIsNotNone(again.get_item(item_key(REPO, 1)))

    def test_upsert_reports_new_updated_unchanged(self) -> None:
        item = _item(1, "crash on login")
        self.assertEqual(self.store.upsert_item(item), "new")
        self.assertEqual(self.store.upsert_item(item), "unchanged")
        changed = _item(1, "crash on login", "now with a body")
        self.assertEqual(self.store.upsert_item(changed), "updated")

    def test_decision_add_get_and_write_audit(self) -> None:
        dec = self.store.add_decision({
            "kind": "duplicate", "question": "Keep #1?",
            "options": ["keep-1", "not-duplicates"],
            "status": "open"})
        self.assertTrue(dec["id"].startswith("dec-"))
        self.assertEqual(self.store.get_decision(dec["id"])["kind"],
                         "duplicate")
        self.assertFalse(self.store.already_written("w-1"))
        self.store.record_write({"key": "w-1", "what": "comment"})
        self.assertTrue(self.store.already_written("w-1"))


if __name__ == "__main__":
    unittest.main()


class _EngineCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TriageStore(Path(self._tmp.name) / "triage.json")
        self.github = FakeGitHub()
        self.store.track(REPO)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _engine(self, suggest: Any = None):
        from nomorals.triage.engine import TriageEngine
        return TriageEngine(self.github, self.store, suggest=suggest)


class TestEngineSync(_EngineCase):
    def test_sync_mirrors_and_counts(self) -> None:
        self.github.seed(1, "crash on login", "segfault on start")
        self.github.seed(2, "add dark mode", "please add a theme")
        engine = self._engine()
        out = engine.sync(REPO)
        self.assertEqual((out["new"], out["updated"], out["unchanged"]),
                         (2, 0, 0))
        out2 = engine.sync(REPO)
        self.assertEqual((out2["new"], out2["updated"], out2["unchanged"]),
                         (0, 0, 2))

    def test_sync_marks_body_change_updated(self) -> None:
        self.github.seed(1, "crash on login", "v1 body")
        engine = self._engine()
        engine.sync(REPO)
        self.github.issues[(REPO, 1)]["body"] = "v2 body with a stack trace"
        out = engine.sync(REPO)
        self.assertEqual(out["updated"], 1)

    def test_sync_failure_is_reported_not_raised(self) -> None:
        self.github.fail_list = "rate limited"
        engine = self._engine()
        out = engine.sync(REPO)
        self.assertEqual(out["new"], 0)
        self.assertIn("rate limited", out["error"])


class TestEngineDuplicates(_EngineCase):
    def test_model_duplicate_verdict_forms_group_and_decision(self) -> None:
        self.github.seed(1, "app crashes on login with valid password")
        self.github.seed(2, "app crashes when logging in with valid password")
        self.github.seed(3, "add dark mode toggle")
        suggest = FakeSuggest({
            "duplicate": json.dumps({
                "verdict": "duplicate", "confidence": 0.95,
                "reason": "same crash, same steps"}),
        })
        engine = self._engine(suggest=suggest)
        engine.sync(REPO)
        groups = engine.find_duplicates(REPO)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]["members"]),
                         [item_key(REPO, 1), item_key(REPO, 2)])
        self.assertEqual(groups[0]["keep"], item_key(REPO, 1))
        decisions = self.store.decisions(status="open")
        self.assertEqual(len(decisions), 1)
        dec = decisions[0]
        self.assertEqual(dec["kind"], "duplicate")
        self.assertIn("Keep #1", dec["question"])
        self.assertIn("not-duplicates", dec["options"])
        # The exact comment texts are stored at ask time.
        self.assertIn("Closing as a duplicate of #1.",
                      dec["texts"][item_key(REPO, 2)])

    def test_model_distinct_verdict_forms_no_group(self) -> None:
        self.github.seed(1, "app crashes on login with valid password")
        self.github.seed(2, "app crashes when logging in with valid password")
        suggest = FakeSuggest({
            "duplicate": json.dumps({
                "verdict": "distinct", "confidence": 0.9,
                "reason": "different stacks"}),
        })
        engine = self._engine(suggest=suggest)
        engine.sync(REPO)
        self.assertEqual(engine.find_duplicates(REPO), [])
        self.assertEqual(self.store.decisions(), [])

    def test_no_model_records_candidates_but_forms_no_group(self) -> None:
        self.github.seed(1, "app crashes on login with valid password")
        self.github.seed(2, "app crashes when logging in with valid password")
        engine = self._engine(suggest=None)
        engine.sync(REPO)
        self.assertEqual(engine.find_duplicates(REPO), [])
        candidates = self.store.candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["status"], "awaiting-model")

    def test_ruled_not_duplicates_are_never_reasked(self) -> None:
        self.github.seed(1, "app crashes on login with valid password")
        self.github.seed(2, "app crashes when logging in with valid password")
        suggest = FakeSuggest({
            "duplicate": json.dumps({
                "verdict": "duplicate", "confidence": 0.95,
                "reason": "same crash"}),
        })
        engine = self._engine(suggest=suggest)
        engine.sync(REPO)
        engine.find_duplicates(REPO)
        dec = self.store.decisions(status="open")[0]
        engine.answer_decision(dec["id"], "not-duplicates")
        # A second pass must not form the group again.
        self.assertEqual(engine.find_duplicates(REPO), [])
        self.assertEqual(self.store.decisions(status="open"), [])


class TestEngineClassifyPrioritize(_EngineCase):
    def test_confident_offline_crash_settles_without_model(self) -> None:
        self.github.seed(1, "App crashes on startup",
                         "segfault immediately, data lost")
        engine = self._engine(suggest=None)
        engine.sync(REPO)
        out = engine.classify_item(item_key(REPO, 1))
        self.assertEqual((out["type"], out["source"]),
                         ("bugfix", "offline"))
        self.assertEqual(out["priority"], "P0")

    def test_unsure_without_model_stays_unsettled_and_needs_decision(self) -> None:
        self.github.seed(1, "Something feels off", "idk")
        engine = self._engine(suggest=None)
        engine.sync(REPO)
        out = engine.classify_item(item_key(REPO, 1))
        self.assertIsNone(out["type"])
        self.assertTrue(out["needs_decision"])
        self.assertEqual(out["priority"], "P2")  # unrated counts as P2

    def test_model_second_opinion_settles_unsure_item(self) -> None:
        self.github.seed(1, "Add support for dark mode",
                         "please add a dark theme option")
        suggest = FakeSuggest({
            "classify": json.dumps({
                "type": "missing feature", "type_confidence": 0.85,
                "security": False, "needs_decision": False,
                "reason": "a new capability ask"}),
        })
        engine = self._engine(suggest=suggest)
        engine.sync(REPO)
        out = engine.classify_item(item_key(REPO, 1))
        self.assertEqual((out["type"], out["source"]),
                         ("missing feature", "model"))
        self.assertEqual(out["priority"], "P2")

    def test_security_flag_raises_priority_one_level(self) -> None:
        self.github.seed(1, "SQL injection in search",
                         "unsanitized input reaches the query")
        engine = self._engine(suggest=None)
        engine.sync(REPO)
        out = engine.classify_item(item_key(REPO, 1))
        self.assertTrue(out["security"])
        self.assertEqual(out["priority"], "P1")

    def test_triage_end_to_end_and_queue_order(self) -> None:
        self.github.seed(1, "App crashes on startup", "segfault, data lost")
        self.github.seed(2, "Typo in README", "spelling mistake in docs")
        self.github.seed(3, "Add support for dark mode", "please add theme")
        suggest = FakeSuggest({
            "Typo": json.dumps({
                "type": "small extra", "type_confidence": 0.9,
                "security": False, "needs_decision": False,
                "reason": "a docs spelling fix"}),
            "classify": json.dumps({
                "type": "missing feature", "type_confidence": 0.85,
                "security": False, "needs_decision": False,
                "reason": "a new capability ask"}),
        })
        engine = self._engine(suggest=suggest)
        out = engine.triage(REPO)
        self.assertEqual(out["synced"]["new"], 3)
        queue = engine.queue(REPO)
        self.assertEqual([q["number"] for q in queue], [1, 3, 2])
        self.assertEqual(queue[0]["priority"], "P0")
        self.assertEqual(queue[2]["priority"], "P3")


class TestEngineApply(_EngineCase):
    def _duplicate_decision(self):
        self.github.seed(1, "app crashes on login with valid password")
        self.github.seed(2, "app crashes when logging in with valid password")
        suggest = FakeSuggest({
            "duplicate": json.dumps({
                "verdict": "duplicate", "confidence": 0.95,
                "reason": "same crash"}),
        })
        engine = self._engine(suggest=suggest)
        engine.sync(REPO)
        engine.find_duplicates(REPO)
        return engine, self.store.decisions(status="open")[0]

    def test_apply_duplicate_posts_texts_and_closes_once(self) -> None:
        engine, dec = self._duplicate_decision()
        engine.answer_decision(dec["id"], "keep-1")
        out = engine.apply_decision(dec["id"])
        self.assertEqual(out["status"], "applied")
        comments = self.github.comments[(REPO, 2)]
        self.assertEqual(len(comments), 1)
        self.assertIn("Closing as a duplicate of #1.", comments[0]["body"])
        self.assertEqual(self.github.issues[(REPO, 2)]["state"], "closed")
        # A second apply is a no-op: the write audit stops repeats.
        out2 = engine.apply_decision(dec["id"])
        self.assertEqual(out2["status"], "already-applied")
        self.assertEqual(len(self.github.comments[(REPO, 2)]), 1)

    def test_apply_scope_writes_the_type_label(self) -> None:
        self.github.seed(1, "Add support for dark mode", "please add theme")
        suggest = FakeSuggest({
            "classify": json.dumps({
                "type": "missing feature", "type_confidence": 0.85,
                "security": False, "needs_decision": False,
                "reason": "a new capability ask"}),
        })
        engine = self._engine(suggest=suggest)
        engine.sync(REPO)
        engine.classify_item(item_key(REPO, 1))
        dec = self.store.add_decision({
            "kind": "scope", "item": item_key(REPO, 1),
            "question": "Is this in scope as a missing feature?",
            "options": ["take", "small-extra", "plugin", "close",
                        "backlog"],
            "pick": "missing feature",
            "status": "open"})
        engine.answer_decision(dec["id"], "take")
        out = engine.apply_decision(dec["id"])
        self.assertEqual(out["status"], "applied")
        self.assertIn((REPO, 1, ["missing feature"]),
                      self.github.label_calls)

    def test_answer_with_unknown_option_is_refused(self) -> None:
        engine, dec = self._duplicate_decision()
        with self.assertRaises(ValueError):
            engine.answer_decision(dec["id"], "explode")


class TestConnectorLabels(unittest.TestCase):
    def test_set_issue_labels_patches_labels(self) -> None:
        from nomorals.connectors.github import GitHubConnector
        calls: list[tuple[str, str, dict]] = []

        def _fake_api(method: str, path: str,
                      payload: dict | None = None, **kw: Any) -> dict:
            calls.append((method, path, payload or {}))
            return {"number": 7, "title": "t", "state": "open",
                    "labels": [{"name": "bugfix"}], "user": {"login": "x"},
                    "comments": 0, "created_at": "", "updated_at": "",
                    "html_url": ""}

        conn = object.__new__(GitHubConnector)
        conn._api = _fake_api  # type: ignore[method-assign]
        out = conn.set_issue_labels(REPO, 7, ["bugfix"])
        self.assertEqual(calls[0][0], "PATCH")
        self.assertIn("/issues/7", calls[0][1])
        self.assertEqual(calls[0][2], {"labels": ["bugfix"]})
        self.assertEqual(out["labels"], ["bugfix"])


class TestToolAndVerb(_EngineCase):
    def _registry(self, suggest: Any = None):
        from nomorals.tools import registry as tool_registry
        from nomorals.tools import triage as triage_tool
        reg = tool_registry.ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home_path=Path(self._tmp.name)),
            _triage_github=self.github,
            _triage_suggest=suggest,
            db=None, router=None))
        triage_tool.register(reg)
        return reg

    def _call(self, reg, **kw: Any) -> dict[str, Any]:
        return reg._tools["triage"].fn(**kw)

    def test_tool_track_sync_queue(self) -> None:
        self.github.seed(1, "App crashes on startup",
                         "segfault immediately, data lost")
        reg = self._registry()
        tracked = self._call(reg, action="track", repo=REPO)
        self.assertTrue(tracked["ok"])
        self.assertIn(REPO, tracked["tracked"])
        out = self._call(reg, action="sync", repo=REPO)
        self.assertTrue(out["ok"])
        self.assertEqual(out["new"], 1)
        queue = self._call(reg, action="queue", repo=REPO)
        self.assertTrue(queue["ok"])
        self.assertEqual(len(queue["items"]), 1)
        self.assertEqual(queue["items"][0]["priority"], "P0")

    def test_tool_without_github_is_honest(self) -> None:
        from nomorals.tools import registry as tool_registry
        from nomorals.tools import triage as triage_tool
        reg = tool_registry.ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home_path=Path(self._tmp.name)),
            db=None, router=None))
        triage_tool.register(reg)
        out = reg._tools["triage"].fn(action="sync", repo=REPO)
        self.assertFalse(out["ok"])
        self.assertIn("not connected", out["error"])

    def test_tool_registers_and_tick_exists(self) -> None:
        reg = self._registry()
        self.assertIsNotNone(reg.get("triage"))
        self.assertIsNotNone(reg.get("__triage_sync__"))

    def test_verb_board_and_queue(self) -> None:
        self.github.seed(1, "App crashes on startup",
                         "segfault immediately, data lost")
        from nomorals.agents.partner.runtime_system import (
            RuntimeSystemMixin,
        )
        runtime = object.__new__(RuntimeSystemMixin)
        runtime._triage_store = lambda: self.store
        runtime._triage_github_client = self.github
        runtime._triage_suggest = lambda: None
        board = runtime._control_triage("", chat_key="telegram:1")
        self.assertIn(REPO, board)
        self.store.track(REPO)
        from nomorals.triage.engine import TriageEngine
        TriageEngine(self.github, self.store, suggest=None).sync(REPO)
        queue = runtime._control_triage("queue", chat_key="telegram:1")
        self.assertIn("#1", queue)
        self.assertIn("P0", queue)


class TestSurface(unittest.TestCase):
    def test_board_names_repo_counts_and_queue(self) -> None:
        from nomorals.triage import surface as triage_surface
        with tempfile.TemporaryDirectory() as tmp:
            store = TriageStore(Path(tmp) / "triage.json")
            store.track(REPO)
            store.upsert_item(_item(1, "crash on login"))
            board = triage_surface.render_board(store)
            self.assertIn(REPO, board)
            self.assertIn("1 open", board)
