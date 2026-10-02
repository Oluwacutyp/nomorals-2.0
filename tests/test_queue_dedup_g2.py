"""Wave G2: propose-time dedup + conflict notes for the upgrade queue.

Covers: near-duplicate merge (provenance kept, files/tests unioned),
genuine conflict -> conflict notes on both sides with no merge,
non-duplicates untouched, and idempotency (re-filing doesn't double-merge).
"""
import unittest
from types import SimpleNamespace

from nomorals.agents.upgrade_chat import (
    render_upgrade_list,
    render_upgrade_show,
)
from nomorals.agents.upgrade_queue import UpgradeQueue
from nomorals.storage.db import Database


def make_context():
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db)


def kw(**overrides):
    base = dict(
        title="Add exponential backoff to market data retries",
        rationale="Market data fetches fail transiently under load; retries "
                "with exponential backoff reduce failed quotes during "
                "volatility spikes across all exchange adapters.",
        patch_plan={"edits": ["wrap fetch in retry loop"], "budget": 2},
        files=["nomorals/integrations/market_data.py"],
        tests=["retry succeeds after two transient failures"],
        claim_ids=["c1"],
        source="research",
    )
    base.update(overrides)
    return base


class DedupTests(unittest.TestCase):
    def setUp(self):
        self.queue = UpgradeQueue(make_context())

    def test_near_duplicate_merges_into_original(self):
        a = self.queue.propose(**kw())
        b = self.queue.propose(**kw(
            title="Add exponential backoff for market data fetch retries",
            rationale="Transient failures in market data fetching should be "
                      "retried with exponential backoff so quotes stop "
                      "failing during volatility spikes.",
            tests=["retry succeeds after two transient failures",
                   "backoff delays grow exponentially"],
            source="digest",
        ))
        # the survivor is the ORIGINAL item
        self.assertEqual(b, a)
        got = self.queue.get(a)
        self.assertEqual(got["status"], "proposed")
        merged = got.get("merged_duplicates") or []
        self.assertEqual(len(merged), 1)
        dup = merged[0]
        # full provenance of the merged filing
        self.assertTrue(dup["id"] and dup["id"] != a)
        self.assertEqual(dup["source"], "digest")
        self.assertTrue(dup["merged_at"] > 0)
        self.assertIn("exponential backoff", dup["title"].lower())
        self.assertIn("exponential backoff", dup["rationale"].lower())
        self.assertTrue(dup["fingerprint"])
        # only one open row exists — nothing silently dropped, nothing
        # duplicated in the queue
        self.assertEqual(len(self.queue.list(status="proposed")), 1)

    def test_merge_unions_files_and_tests(self):
        a = self.queue.propose(**kw())
        self.queue.propose(**kw(
            title="Market data fetch retries with exponential backoff",
            rationale="Market data fetches fail transiently under load; "
                      "retries with exponential backoff reduce failed quotes "
                      "during volatility spikes across all adapters.",
            files=["nomorals/integrations/market_data.py",
                   "nomorals/integrations/websocket.py"],
            tests=["retry succeeds after two transient failures"],
            claim_ids=["c9"],
        ))
        got = self.queue.get(a)
        self.assertEqual(got["files"],
                         ["nomorals/integrations/market_data.py",
                          "nomorals/integrations/websocket.py"])
        # no dupes introduced by the union
        self.assertEqual(len(got["files"]), len(set(got["files"])))
        self.assertEqual(got["tests"],
                         ["retry succeeds after two transient failures"])
        self.assertEqual(sorted(got["claim_ids"]), ["c1", "c9"])

    def test_conflict_notes_on_both_sides_no_merge(self):
        a = self.queue.propose(**kw(
            title="Add a caching layer to market data fetches",
            rationale="Market data fetches are expensive; adding a TTL cache "
                      "layer in front of every exchange adapter cuts latency "
                      "and API spend during volatility spikes.",
            files=["nomorals/integrations/market_data.py"],
        ))
        b = self.queue.propose(**kw(
            title="Do not add a caching layer to market data fetches",
            rationale="Do not add a caching layer to market data fetches: "
                      "stale cached quotes are worse than slow fresh ones "
                      "during volatility spikes, so keep every fetch live.",
            files=["nomorals/integrations/market_data.py"],
        ))
        self.assertNotEqual(a, b)
        # both stay open — no merge, no last-write-wins
        self.assertEqual(len(self.queue.list(status="proposed")), 2)
        ga, gb = self.queue.get(a), self.queue.get(b)
        self.assertEqual(ga.get("merged_duplicates"), [])
        self.assertEqual(gb.get("merged_duplicates"), [])
        na = ga.get("conflict_notes") or []
        nb = gb.get("conflict_notes") or []
        self.assertEqual(len(na), 1)
        self.assertEqual(len(nb), 1)
        # each side visible from the other
        self.assertEqual(na[0]["proposal_id"], b)
        self.assertEqual(nb[0]["proposal_id"], a)
        self.assertIn("caching", na[0]["title"].lower())
        self.assertIn("market_data.py", na[0]["point"])
        self.assertTrue(na[0]["noted_at"] > 0)
        self.assertTrue(nb[0]["noted_at"] > 0)

    def test_affirmations_with_same_files_do_not_conflict(self):
        a = self.queue.propose(**kw())
        b = self.queue.propose(**kw(
            title="Exponential backoff wrapper for market data fetches",
            rationale="Market data fetches fail transiently under load, so "
                      "wrap them in retries with exponential backoff to cut "
                      "failed quotes during volatility spikes.",
            files=["nomorals/integrations/market_data.py"],
        ))
        self.assertEqual(b, a)  # merged, not conflicted
        got = self.queue.get(a)
        self.assertEqual(got.get("conflict_notes"), [])

    def test_non_duplicates_untouched(self):
        a = self.queue.propose(**kw())
        b = self.queue.propose(**kw(
            title="Rewrite the arena topic rotation scheduler",
            rationale="Arena topic rotation currently repeats too often; a "
                      "weighted anti-repeat scheduler keeps debates fresh "
                      "for regular players over long sessions.",
            files=["nomorals/games/arena/topics.py"],
            tests=["no topic repeats within ten rounds"],
        ))
        self.assertNotEqual(a, b)
        self.assertEqual(len(self.queue.list(status="proposed")), 2)
        ga, gb = self.queue.get(a), self.queue.get(b)
        self.assertEqual(ga.get("merged_duplicates"), [])
        self.assertEqual(ga.get("conflict_notes"), [])
        self.assertEqual(gb.get("merged_duplicates"), [])
        self.assertEqual(gb.get("conflict_notes"), [])

    def test_idempotent_refile(self):
        a = self.queue.propose(**kw())
        again = self.queue.propose(**kw())
        self.assertEqual(again, a)
        got = self.queue.get(a)
        # exact re-file is a no-op: no merged-duplicate record created
        self.assertEqual(got.get("merged_duplicates"), [])
        self.assertEqual(len(self.queue.list(status="proposed")), 1)

    def test_no_double_merge_of_same_duplicate(self):
        a = self.queue.propose(**kw())
        dup_kwargs = kw(
            title="Add exponential backoff for market data fetch retries",
            rationale="Transient failures in market data fetching should be "
                      "retried with exponential backoff so quotes stop "
                      "failing during volatility spikes.",
            tests=["retry succeeds after two transient failures",
                   "backoff delays grow exponentially"],
        )
        self.queue.propose(**dup_kwargs)
        self.queue.propose(**dup_kwargs)  # same duplicate filed twice
        got = self.queue.get(a)
        self.assertEqual(len(got.get("merged_duplicates") or []), 1)
        self.assertEqual(len(self.queue.list(status="proposed")), 1)

    def test_merge_keeps_original_patch_plan_and_source(self):
        a = self.queue.propose(**kw())
        self.queue.propose(**kw(
            title="Market data fetch retries with exponential backoff",
            rationale="Market data fetches fail transiently under load; "
                      "retries with exponential backoff reduce failed quotes "
                      "during volatility spikes across all adapters.",
            patch_plan={"edits": ["something else entirely"], "budget": 9},
            source="other",
        ))
        got = self.queue.get(a)
        # the original plan wins; the duplicate's plan does not overwrite it
        self.assertEqual(got["patch_plan"]["budget"], 2)
        self.assertEqual(got["source"], "research")

    def test_decided_proposals_not_deduped(self):
        a = self.queue.propose(**kw())
        self.queue.deny(a, "not now", by="owner")
        b = self.queue.propose(**kw(
            title="Market data fetch retries with exponential backoff",
            rationale="Market data fetches fail transiently under load; "
                      "retries with exponential backoff reduce failed quotes "
                      "during volatility spikes across all adapters.",
        ))
        self.assertNotEqual(b, a)
        self.assertEqual(len(self.queue.list(status="proposed")), 1)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.queue = UpgradeQueue(make_context())

    def test_list_flags_merged_and_conflicts(self):
        a = self.queue.propose(**kw())
        self.queue.propose(**kw(
            title="Market data fetch retries with exponential backoff",
            rationale="Market data fetches fail transiently under load; "
                      "retries with exponential backoff reduce failed quotes "
                      "during volatility spikes across all adapters.",
        ))
        text = render_upgrade_list(self.queue.list(status="proposed"))
        self.assertIn("+1 merged", text)

        c = self.queue.propose(**kw(
            title="Do not retry market data fetches at all",
            rationale="Do not retry market data fetches: transient failures "
                      "during volatility spikes should surface immediately "
                      "instead of hiding behind exponential backoff loops.",
            files=["nomorals/integrations/market_data.py"],
        ))
        text = render_upgrade_list(self.queue.list(status="proposed"))
        self.assertIn("1 conflict(s)", text)
        self.assertIn(c, text)

    def test_show_renders_merged_and_conflicts(self):
        a = self.queue.propose(**kw())
        self.queue.propose(**kw(
            title="Market data fetch retries with exponential backoff",
            rationale="Market data fetches fail transiently under load; "
                      "retries with exponential backoff reduce failed quotes "
                      "during volatility spikes across all adapters.",
            source="digest",
        ))
        c = self.queue.propose(**kw(
            title="Do not retry market data fetches at all",
            rationale="Do not retry market data fetches: transient failures "
                      "during volatility spikes should surface immediately "
                      "instead of hiding behind exponential backoff loops.",
            files=["nomorals/integrations/market_data.py"],
        ))
        text = render_upgrade_show(self.queue.get(a))
        self.assertIn("merged duplicates (1):", text)
        self.assertIn("[digest]", text)
        self.assertIn("conflict notes (1)", text)
        self.assertIn(c, text)
        self.assertIn("market_data.py", text)


if __name__ == "__main__":
    unittest.main()
