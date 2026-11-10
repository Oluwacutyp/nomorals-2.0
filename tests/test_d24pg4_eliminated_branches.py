"""Doc 24 PG4 — eliminated-branch preservation.

PG1–PG3 gave every claim a grade derived from its evidence.
PG4 closes the preservation gap the evaluation named: branches
that LOSE — a claim superseded by re-evaluation, a claim whose
candidate is shed, a niche-discovery cluster the evidence bar
declines — used to vanish as records (supersede left an unmarked
history row; niche declines lived only in a per-run report).

Pins:

(a) a superseding re-evaluation preserves the prior version as
    an eliminated branch: full claim snapshot, its OWN evidence
    refs + derived grade, the reason (the material fields that
    changed), and the superseding claim — while the live set
    carries only the new version;
(b) a shed candidate's claim is preserved with the shed verdict's
    reason and no winner (superseded_by is None);
(c) niche-discovery declines are preserved with the cluster,
    its evidence snapshot and the reason — a loser with bigger
    raw sighting numbers but weaker evidence grades BELOW a
    corroborated loser: grades follow evidence, not outcomes
    and not counts;
(d) preservation is not resurrection: eliminated branches never
    appear in current claims, in the claims feed, or among the
    live claims on the rendered surface (they render in their
    own labelled section);
(e) re-derivation honesty: a fiat grade token planted on a
    preserved record reads back at the grade its evidence refs
    earn; a branch with only a generative-thought ref reads G0
    and renders as G0 in the eliminated section;
(f) idempotency + persistence: re-running the evaluator or
    discovery over unchanged evidence preserves nothing twice,
    and the records survive a store reload through the KV idiom.
"""

from __future__ import annotations

import unittest

from nomorals.media.explore import (
    ExploreStore,
    current_claims,
    discover_niches,
    eliminated_branches,
    evaluate_trends,
)
from nomorals.media.explore.board import (
    render_explore_niches,
    render_explore_trends,
)
from nomorals.media.explore.feed import claims_as_findings
from nomorals.media.explore.trends import proof_grade_for_branch

T0 = 1_700_000_000.0


# ── seeding helpers (the shipped seams, no back doors) ──────────


def _row(rank, title, url):
    return {"rank": rank, "title": title, "url": url,
            "source": "youtube", "channel": None,
            "view_count": None, "duration_s": None}


def _rows(n, tag):
    return [_row(i + 1, f"{tag} video {i + 1}",
                 f"https://youtu.be/{tag}{i}") for i in range(n)]


def _sweep(store, beats, at, dur=10.0):
    sweep_id = store.begin_sweep(trigger="test", beats=beats, at=at)
    store.finish_sweep(sweep_id, at=at + dur)
    return sweep_id


def _obs(store, sweep_id, beat, instrument, rows, at):
    status = "ok" if rows else "empty"
    return store.record_observation({
        "sweep_id": sweep_id, "beat": beat,
        "instrument": instrument, "query": beat, "at": at,
        "status": status, "error": "",
        "result_count": len(rows), "results": rows})


def _spike_then_trend_store():
    """One beat: a single-sweep burst (spike), then a second
    corroborating sweep — the re-evaluation supersedes the spike
    with a trend."""
    store = ExploreStore()
    s1 = _sweep(store, ["retro gaming"], at=T0)
    _obs(store, s1, "retro gaming", "web_search", [], at=T0 + 1)
    s2 = _sweep(store, ["retro gaming"], at=T0 + 1000)
    _obs(store, s2, "retro gaming", "web_search",
         _rows(6, "retro"), at=T0 + 1001)
    first = evaluate_trends(store, now=T0 + 2000)
    s3 = _sweep(store, ["retro gaming"], at=T0 + 3000)
    _obs(store, s3, "retro gaming", "youtube_search",
         _rows(5, "retro2"), at=T0 + 3001)
    second = evaluate_trends(store, now=T0 + 4000)
    return store, first, second


def _note(store, kind, name, *, beat, sweep_id, url, now,
          instrument="youtube_search"):
    return store.note_candidate(
        kind, name, via=f"sweep:{instrument}",
        platform="youtube" if kind == "channel" else "",
        beat=beat,
        evidence={"title": f"{name} — a video", "url": url,
                  "rank": 1, "view_count": 9000,
                  "instrument": instrument, "beat": beat},
        sweep_id=sweep_id, now=now)


class SupersedePreservationTests(unittest.TestCase):

    def test_superseded_claim_is_preserved_with_reason_and_winner(self):
        store, first, second = _spike_then_trend_store()
        self.assertEqual(first["eliminated_preserved"], 0)
        spike = first["claims_current"][0]
        self.assertEqual(spike["kind"], "spike")

        self.assertEqual(second["eliminated_preserved"], 1)
        live = second["claims_current"]
        self.assertEqual([c["kind"] for c in live], ["trend"])

        branches = eliminated_branches(store)
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        self.assertEqual(branch["branch_kind"], "claim")
        self.assertEqual(branch["outcome"], "superseded")
        self.assertTrue(branch["eliminated"])
        # the preserved branch IS the old version, whole
        self.assertEqual(branch["claim"]["kind"], "spike")
        self.assertEqual(branch["claim"]["text"], spike["text"])
        self.assertEqual(branch["text"], spike["text"])
        # the reason names what changed; the winner is linked
        self.assertIn("kind changed spike → trend",
                      branch["reason"])
        self.assertEqual(branch["superseded_by"]["kind"], "trend")
        self.assertEqual(branch["superseded_by"]["text"],
                         live[0]["text"])
        # its grade derives from ITS evidence (the spike's
        # derivation), not from the trend that replaced it
        self.assertEqual(branch["proof_grade"]["grade"], "G2")
        self.assertTrue(any(
            e["kind"] == "derivation"
            for e in branch["proof_grade"]["evidence"]))
        self.assertEqual(live[0]["proof_grade"]["grade"], "G4")

    def test_preservation_is_additive_history_keeps_both_rows(self):
        store, _first, _second = _spike_then_trend_store()
        kinds = [r["kind"] for r in store.claims(limit=10)]
        self.assertEqual(sorted(kinds), ["spike", "trend"])
        # and the live fold still yields exactly the new version
        self.assertEqual(
            [c["kind"] for c in current_claims(store)], ["trend"])

    def test_reevaluation_is_idempotent_no_duplicate_branch(self):
        store, _first, _second = _spike_then_trend_store()
        third = evaluate_trends(store, now=T0 + 5000)
        self.assertEqual(third["claims_written"], 0)
        self.assertEqual(third["eliminated_preserved"], 0)
        self.assertEqual(len(eliminated_branches(store)), 1)


class ShedPreservationTests(unittest.TestCase):

    def test_shed_candidates_claim_is_preserved_with_shed_reason(self):
        store = ExploreStore()
        _note(store, "topic", "quiet corner", beat="gold trading",
              sweep_id="intake", url="https://youtu.be/q1", now=T0)
        first = evaluate_trends(store, now=T0 + 500)
        key = ExploreStore.candidate_key("topic", "quiet corner")
        claimed = [c for c in first["claims_current"]
                   if c.get("subject") == key]
        self.assertEqual(len(claimed), 1)

        # three completed sweeps cover its beat after its last
        # sighting without sighting it — the absence verdict sheds
        for i in range(3):
            _sweep(store, ["gold trading"], at=T0 + 1000 * (i + 1))
        second = evaluate_trends(store, now=T0 + 4000)
        self.assertEqual(second["candidates"]["shed"], [key])
        self.assertEqual(second["eliminated_preserved"], 1)
        self.assertEqual(
            [c for c in current_claims(store)
             if c.get("subject") == key], [])

        branches = eliminated_branches(store)
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        self.assertEqual(branch["outcome"], "shed")
        self.assertIn("absent from 3 consecutive",
                      branch["reason"])
        # a verdict eliminated it — no winner superseded it
        self.assertIsNone(branch["superseded_by"])
        self.assertEqual(branch["claim"]["text"],
                         claimed[0]["text"])


class FanOutLoserTests(unittest.TestCase):

    def test_single_sweep_burst_loses_and_grades_by_its_evidence(self):
        # six sightings — bigger raw numbers than the corroborated
        # cluster below — but all inside ONE sweep
        store = ExploreStore()
        for url in ("https://youtu.be/a", "https://youtu.be/b",
                    "https://youtu.be/c"):
            _note(store, "channel", "One Sweep TV", beat="one beat",
                  sweep_id="sweep-1", url=url, now=T0)
            _note(store, "topic", "one beat topic", beat="one beat",
                  sweep_id="sweep-1", url=url, now=T0)
        for cand in store.candidates():
            store.update_candidate_verdict(
                cand["key"], status="promoted",
                standing={"sightings": 3, "score": 9})
        report = discover_niches(store, now=T0 + 5000)
        self.assertEqual(report["proposed"], [])
        self.assertTrue(any(
            "single sweep" in d["reason"]
            for d in report["declined"]))

        branches = eliminated_branches(
            store, branch_kind="niche_cluster")
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        self.assertEqual(branch["outcome"], "declined")
        self.assertIn("single sweep", branch["reason"])
        self.assertEqual(branch["evidence"]["sightings"], 6)
        self.assertEqual(
            {m["name"] for m in branch["members"]},
            {"One Sweep TV", "one beat topic"})
        self.assertIsNone(branch["superseded_by"])
        # 6 sightings, one sweep: the ONLY ref its evidence earns
        # is the evaluation filter — G1, however loud the burst
        self.assertEqual(branch["proof_grade"]["grade"], "G1")
        self.assertEqual(
            [e["kind"] for e in branch["proof_grade"]["evidence"]],
            ["logical_check"])

    def test_corroborated_loser_keeps_its_empirical_grade(self):
        # two members across two sweeps, never promoted, no trend
        # claim: declined on standing — but its evidence WAS
        # empirical, and the preserved grade says so
        store = ExploreStore()
        for i, sweep in enumerate(["sweep-1", "sweep-2"]):
            t = T0 + i * 3600
            for url in ("https://youtu.be/g1", "https://youtu.be/g2"):
                _note(store, "channel", "NG Gold Room",
                      beat="gold trading", sweep_id=sweep, url=url,
                      now=t)
                _note(store, "topic", "xauusd level breakdown",
                      beat="gold trading", sweep_id=sweep, url=url,
                      now=t)
        report = discover_niches(store, now=T0 + 5000)
        self.assertEqual(report["proposed"], [])
        self.assertTrue(any(
            "watching evidence only" in d["reason"]
            for d in report["declined"]))
        branches = eliminated_branches(
            store, branch_kind="niche_cluster")
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        self.assertEqual(branch["proof_grade"]["grade"], "G4")
        self.assertTrue(any(
            e["kind"] == "empirical_source"
            for e in branch["proof_grade"]["evidence"]))

    def test_cluster_covered_by_existing_niche_names_its_winner(self):
        store = ExploreStore()
        for i, sweep in enumerate(["sweep-1", "sweep-2"]):
            t = T0 + i * 3600
            for url in ("https://youtu.be/f1", "https://youtu.be/f2"):
                _note(store, "channel", "Money Facts Daily",
                      beat="finance facts", sweep_id=sweep, url=url,
                      now=t)
                _note(store, "topic", "money fact of the day",
                      beat="finance facts", sweep_id=sweep, url=url,
                      now=t)
        for cand in store.candidates():
            store.update_candidate_verdict(
                cand["key"], status="promoted",
                standing={"sightings": 4, "score": 9})
        report = discover_niches(store, now=T0 + 5000)
        self.assertEqual(report["proposed"], [])
        branches = eliminated_branches(
            store, branch_kind="niche_cluster")
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        self.assertEqual(branch["superseded_by"]["kind"], "niche")
        self.assertEqual(branch["superseded_by"]["ref"],
                         "finance_facts")
        # discovery re-run over the same evidence: no duplicate
        discover_niches(store, now=T0 + 6000)
        self.assertEqual(len(eliminated_branches(
            store, branch_kind="niche_cluster")), 1)

    def test_declined_clusters_render_on_the_niches_surface(self):
        store = ExploreStore()
        _note(store, "channel", "Solo Channel", beat="solo beat",
              sweep_id="sweep-1", url="https://youtu.be/y", now=T0)
        store.update_candidate_verdict(
            store.candidates()[0]["key"], status="promoted",
            standing={"sightings": 4, "score": 9})
        discover_niches(store, now=T0 + 100)
        text = render_explore_niches(store=store)
        self.assertIn("declined clusters", text)
        self.assertIn("single member", text)
        self.assertIn("not proposals", text)


class NoResurrectionTests(unittest.TestCase):

    def test_eliminated_never_enters_current_claims_or_the_feed(self):
        store, _first, _second = _spike_then_trend_store()
        self.assertEqual(
            [c["kind"] for c in current_claims(store)], ["trend"])
        findings = claims_as_findings(store, now=T0 + 4000)
        self.assertEqual([f["claim_kind"] for f in findings],
                         ["trend"])
        self.assertFalse(any(
            "spike" in (f.get("title") or "") for f in findings))

    def test_render_labels_eliminated_outside_the_live_claims(self):
        store, first, _second = _spike_then_trend_store()
        spike_text = first["claims_current"][0]["text"]
        text = render_explore_trends(
            store=store, evaluate=False, now=T0 + 4000)
        live_part, _, eliminated_part = text.partition(
            "eliminated branches")
        self.assertNotIn(spike_text, live_part)
        self.assertIn(spike_text, eliminated_part)
        self.assertIn("not live claims", text)
        self.assertIn("kind changed spike → trend",
                      eliminated_part)
        self.assertIn("proof G2 (its own evidence)",
                      eliminated_part)
        # no internal record ids on the surface
        self.assertNotIn("claim:explore", text)


class ReDerivationHonestyTests(unittest.TestCase):

    def test_planted_fiat_grade_reads_back_at_its_evidence(self):
        store = ExploreStore()
        store.record_eliminated_branch({
            "branch_kind": "claim",
            "subject": "planted", "subject_key": "planted",
            "subject_kind": "beat", "kind": "trend",
            "text": "planted branch",
            "claim": {"claim_id": "claim:planted",
                      "recorded_at": 1.0},
            "evidence_refs": [
                {"kind": "derivation", "ref": "claim:planted:x"}],
            "proof_grade": {"grade": "G6", "evidence": []},
            "outcome": "superseded", "reason": "planted",
            "superseded_by": None, "eliminated_at": T0,
            "source": "test",
        })
        branches = eliminated_branches(store)
        self.assertEqual(len(branches), 1)
        # the G6 token is a claim about the record; the record's
        # evidence earns G2, so G2 is what reads back
        self.assertEqual(branches[0]["proof_grade"]["grade"], "G2")

    def test_thought_only_branch_is_g0_and_renders_as_g0(self):
        store = ExploreStore()
        store.record_eliminated_branch({
            "branch_kind": "claim",
            "subject": "hunch", "subject_key": "hunch",
            "subject_kind": "beat", "kind": "unverified",
            "text": "a hunch with no legs",
            "claim": {"claim_id": "claim:hunch",
                      "recorded_at": 2.0},
            "evidence_refs": [
                {"kind": "generative_thought",
                 "ref": "claim:hunch"}],
            "outcome": "superseded", "reason": "planted",
            "superseded_by": None, "eliminated_at": T0,
            "source": "test",
        })
        branch = eliminated_branches(store)[0]
        self.assertEqual(
            proof_grade_for_branch(branch).grade.name, "G0")
        text = render_explore_trends(
            store=store, evaluate=False, now=T0)
        self.assertIn("a hunch with no legs", text)
        self.assertIn("proof G0 (its own evidence)", text)


class PersistenceTests(unittest.TestCase):

    def test_branches_survive_a_store_reload(self):
        class _KV:
            def __init__(self):
                self.data = {}

            def get(self, key):
                return self.data.get(key)

            def set(self, key, value):
                self.data[key] = value

        kv = _KV()
        store = ExploreStore(kv)
        s1 = _sweep(store, ["retro gaming"], at=T0)
        _obs(store, s1, "retro gaming", "web_search", [],
             at=T0 + 1)
        s2 = _sweep(store, ["retro gaming"], at=T0 + 1000)
        _obs(store, s2, "retro gaming", "web_search",
             _rows(6, "retro"), at=T0 + 1001)
        evaluate_trends(store, now=T0 + 2000)
        s3 = _sweep(store, ["retro gaming"], at=T0 + 3000)
        _obs(store, s3, "retro gaming", "youtube_search",
             _rows(5, "retro2"), at=T0 + 3001)
        evaluate_trends(store, now=T0 + 4000)

        reloaded = ExploreStore(kv)
        branches = eliminated_branches(reloaded)
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]["claim"]["kind"], "spike")
        self.assertEqual(branches[0]["proof_grade"]["grade"], "G2")
        self.assertEqual(
            reloaded.status()["eliminated_branches"], 1)


if __name__ == "__main__":
    unittest.main()
