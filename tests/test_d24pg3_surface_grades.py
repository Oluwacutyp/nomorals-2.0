"""Doc 24 PG3 — proof grades EMITTED at the claim surfaces.

PG1 (``core/proof_grade.py``) defined the G0–G6 scale and the
WS4 kind mapping; PG2 enforced promotion in the impossible
ledger. This pass wires emission at the two remaining claim
surfaces:

* ``agents/research_digest`` — every claim carries a ClaimGrade
  reading derived from the sources IT cites (never from its
  confidence number), on its durable record (``to_dict``, the KG
  node written by ``promote``), in ``promote``'s rejected
  entries, and as a visible mark in the brief + note renders.
* ``media/explore/trends`` (WS4) — every claim carries a
  ``proof_grade`` derived from the evidence legs ATTACHED to
  its record (sweeps / corroboration / derivation), stamped at
  authoring, re-derived at read, rendered on the board + the
  trends text, and carried into the WS3 feed's findings.

Pins:

(a) digest discrimination: multi-source > no-source, ordered by
    the machinery; a 0.99-confidence claim with NO sources grades
    exactly like a 0.10 one with none (confidence never grades);
(b) digest durable + render: to_dict carries grade + evidence
    refs + basis; promote writes the grade onto the KG node and
    onto rejected entries; brief and note show the grade mark
    for a sourced AND an unsourced finding;
(c) WS4 authored claims grade AT the PG1 mapping for their kind
    (trend G4 / spike G2 / unverified G1) — asserted against
    ``grade_for_ws4_kind`` itself;
(d) WS4 label-only demotion: a hand-stored row LABELLED trend
    whose basis lacks the corroboration legs grades below G4;
    the same row WITH the legs grades G4;
(e) velocity never upgrades: an unverified claim carrying an
    extreme velocity dict stays G1; a spike's velocity earns G2
    only for the spike kind;
(f) WS4 records + renders: stored rows carry the stamp, the
    trends render and the board show "proof G4", the feed
    finding carries the grade;
(g) import route (no mirror): both organs' grades ARE
    ``core.proof_grade`` objects — media is L4 and agents L5,
    both above core L1, so PG2's mirror discipline does not
    apply and no mirrored table exists to drift.
"""

from __future__ import annotations

import unittest

from nomorals.core.proof_grade import (
    ClaimGrade,
    ProofGrade,
    grade_for_ws4_kind,
)

# ── (a)+(b) research_digest ─────────────────────────────────────


def _digest_claim(cid, text, *, sources=(), confidence=0.3):
    from nomorals.agents.research_digest import ResearchClaim

    return ResearchClaim(
        id=cid, claim=text,
        sources=[{"url": u, "title": "", "trust": 0.6}
                 for u in sources],
        confidence=confidence)


class DigestGradeTests(unittest.TestCase):

    def test_multi_source_grades_above_no_source(self):
        from nomorals.agents.research_digest import claim_proof_grade

        backed = _digest_claim(
            "pg3-a", "Postgres outperforms MySQL for analytical "
                     "workloads on large datasets",
            sources=("https://a.example/1", "https://b.example/2",
                     "https://c.example/3"), confidence=0.99)
        bare = _digest_claim(
            "pg3-b", "Postgres outperforms MySQL for analytical "
                     "workloads on large datasets",
            sources=(), confidence=0.99)
        self.assertEqual(claim_proof_grade(backed).grade,
                         ProofGrade.G4)
        self.assertEqual(claim_proof_grade(bare).grade,
                         ProofGrade.G1)
        self.assertGreater(claim_proof_grade(backed).grade,
                           claim_proof_grade(bare).grade)

    def test_confidence_never_grades(self):
        from nomorals.agents.research_digest import claim_proof_grade

        text = ("Postgres outperforms MySQL for analytical "
                "workloads on large datasets")
        loud = _digest_claim("pg3-c", text, confidence=0.99)
        quiet = _digest_claim("pg3-d", text, confidence=0.10)
        self.assertEqual(claim_proof_grade(loud).grade,
                         claim_proof_grade(quiet).grade)
        sourced_quiet = _digest_claim(
            "pg3-e", text, sources=("https://a.example/1",),
            confidence=0.10)
        self.assertEqual(claim_proof_grade(sourced_quiet).grade,
                         ProofGrade.G4)
        self.assertGreater(claim_proof_grade(sourced_quiet).grade,
                           claim_proof_grade(loud).grade)

    def test_thin_unsourced_claim_is_g0(self):
        from nomorals.agents.research_digest import claim_proof_grade

        thin = _digest_claim("pg3-f", "big news", confidence=0.99)
        self.assertEqual(claim_proof_grade(thin).grade,
                         ProofGrade.G0)

    def test_duplicate_urls_count_once(self):
        from nomorals.agents.research_digest import claim_proof_grade, proof_grade_payload

        dup = _digest_claim(
            "pg3-g", "Postgres outperforms MySQL for analytical "
                     "workloads on large datasets",
            sources=("https://a.example/1", "https://a.example/1"))
        payload = proof_grade_payload(dup)
        self.assertEqual(len(payload["evidence"]), 1)
        self.assertIn("1 cited source", payload["basis"])
        self.assertEqual(claim_proof_grade(dup).grade,
                         ProofGrade.G4)

    def test_to_dict_carries_grade_and_round_trips(self):
        from nomorals.agents.research_digest import ResearchClaim

        claim = _digest_claim(
            "pg3-h", "Postgres outperforms MySQL for analytical "
                     "workloads on large datasets",
            sources=("https://a.example/1", "https://b.example/2"))
        data = claim.to_dict()
        self.assertEqual(data["proof_grade"]["grade"], "G4")
        self.assertEqual(len(data["proof_grade"]["evidence"]), 2)
        self.assertEqual(data["proof_grade"]["evidence"][0]["kind"],
                         "empirical_source")
        self.assertIn("corroborated", data["proof_grade"]["basis"])
        back = ResearchClaim.from_dict(data)
        self.assertEqual(back.to_dict(), data)

    def test_promote_writes_grade_to_node_and_rejected(self):
        from unittest import mock

        from nomorals.agents.research_digest import promote

        class _Node:
            def __init__(self, nid):
                self.id = nid

        class _KG:
            def __init__(self, db):
                self.nodes = {}
                self.links = []

            def recall(self, text, limit=12):
                return []

            def upsert_node(self, label, type=None, properties=None):
                self.nodes[label] = dict(properties or {})
                return _Node(label)

            def link(self, a, b, kind, weight=1.0):
                self.links.append((a, b, kind))

        planted = {}

        def _factory(db):
            kg = _KG(db)
            planted["kg"] = kg
            return kg

        good = _digest_claim(
            "pg3-i", "Postgres outperforms MySQL for analytical "
                     "workloads on large datasets",
            sources=("https://a.example/1",), confidence=0.9)
        weak = _digest_claim(
            "pg3-j", "Postgres outperforms MySQL for analytical "
                     "workloads on large datasets",
            sources=(), confidence=0.1)
        with mock.patch(
                "nomorals.agents.research_digest.KnowledgeGraph",
                _factory):
            result = promote([good, weak], db=object())
        self.assertEqual(result["promoted"], 1)
        node = planted["kg"].nodes["claim:pg3-i"]
        self.assertEqual(node["proof_grade"], "G4")
        self.assertEqual(len(result["rejected"]), 1)
        rejected = result["rejected"][0]
        self.assertEqual(rejected["claim_id"], "pg3-j")
        self.assertEqual(rejected["proof_grade"]["grade"], "G1")
        self.assertTrue(rejected["reasons"])

    def test_renders_show_grade_marks(self):
        from nomorals.agents.research_digest import ResearchClaim, ResearchDigest
        from nomorals.agents.research_swarm import SwarmFinding, SwarmReport

        sourced = SwarmFinding(
            angle="a1", claim="Postgres scales analytics better",
            sources=[{"url": "https://a.example/1", "title": "A",
                      "trust": 0.7},
                     {"url": "https://b.example/2", "title": "B",
                      "trust": 0.7}],
            confidence=0.8)
        bare = SwarmFinding(
            angle="a2", claim="MySQL surely wins everywhere",
            sources=[], confidence=0.8)
        report = SwarmReport(query="db analytics",
                             angles=["a1", "a2"],
                             findings=[sourced, bare])
        brief = ResearchDigest.operator_brief(report)
        self.assertIn("· G4]", brief)
        self.assertIn("· G1]", brief)
        note = ResearchDigest.technical_note(report)
        self.assertIn("· G4]", note)
        self.assertIn("· G1]", note)
        # ResearchClaim inputs render through the same marks
        claims = [
            ResearchClaim(
                id="pg3-k", claim="Postgres scales analytics better",
                sources=[{"url": "https://a.example/1"}],
                confidence=0.9),
            ResearchClaim(
                id="pg3-l", claim="MySQL surely wins everywhere",
                sources=[], confidence=0.9),
        ]
        brief2 = ResearchDigest.operator_brief(claims)
        self.assertIn("· G4]", brief2)
        self.assertIn("· G1]", brief2)


# ── (c)–(f) WS4 exploration claims ──────────────────────────────


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


def _claim_for(claims, subject):
    for claim in claims:
        if claim.get("subject") == subject:
            return claim
    return None


def _trend_store():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    s1 = _sweep(store, ["gold trading"], at=1000.0)
    _obs(store, s1, "gold trading", "youtube_search",
         _rows(3, "g1"), at=1001.0)
    s2 = _sweep(store, ["gold trading"], at=2000.0)
    _obs(store, s2, "gold trading", "web_search",
         _rows(4, "g3"), at=2001.0)
    report = evaluate_trends(store, now=2100.0)
    return store, report


class ExploreGradeTests(unittest.TestCase):

    def test_authored_kinds_grade_at_the_pg1_mapping(self):
        from nomorals.media.explore import ExploreStore, evaluate_trends

        _store, report = _trend_store()
        trend = _claim_for(report["claims_current"], "gold trading")
        self.assertEqual(trend["kind"], "trend")
        self.assertEqual(trend["proof_grade"]["grade"], "G4")
        self.assertEqual(
            trend["proof_grade"]["grade"],
            grade_for_ws4_kind("trend").name)
        self.assertTrue(any(
            e["kind"] == "empirical_source"
            for e in trend["proof_grade"]["evidence"]))

        store = ExploreStore()
        s1 = _sweep(store, ["retro gaming"], at=100.0)
        _obs(store, s1, "retro gaming", "web_search", [], at=101.0)
        s2 = _sweep(store, ["retro gaming"], at=200.0)
        _obs(store, s2, "retro gaming", "web_search",
             _rows(6, "retro"), at=201.0)
        s3 = _sweep(store, ["home baking"], at=300.0)
        _obs(store, s3, "home baking", "youtube_search",
             _rows(2, "bake"), at=301.0)
        report = evaluate_trends(store, now=400.0)
        spike = _claim_for(report["claims_current"], "retro gaming")
        self.assertEqual(spike["kind"], "spike")
        self.assertEqual(spike["proof_grade"]["grade"], "G2")
        self.assertEqual(spike["proof_grade"]["grade"],
                         grade_for_ws4_kind("spike").name)
        unv = _claim_for(report["claims_current"], "home baking")
        self.assertEqual(unv["kind"], "unverified")
        self.assertEqual(unv["proof_grade"]["grade"], "G1")
        self.assertEqual(unv["proof_grade"]["grade"],
                         grade_for_ws4_kind("unverified").name)

    def test_candidate_spike_grades_g2_from_sighting_count(self):
        from nomorals.media.explore import ExploreStore, evaluate_trends

        store = ExploreStore()
        s1 = _sweep(store, ["gold trading"], at=100.0)
        for i in range(3):
            store.note_candidate(
                "channel", "Burst Desk", via="sweep:youtube_search",
                platform="youtube", beat="gold trading",
                sweep_id=s1, now=101.0 + i,
                evidence={"title": "Levels",
                          "url": f"https://youtu.be/b{i}",
                          "instrument": "youtube_search",
                          "beat": "gold trading"})
        report = evaluate_trends(store, now=200.0)
        key = ExploreStore.candidate_key(
            "channel", "Burst Desk", "youtube")
        claim = _claim_for(report["claims_current"], key)
        self.assertIsNotNone(claim)
        self.assertEqual(claim["kind"], "spike")
        self.assertEqual(claim["basis"]["sightings"], 3)
        self.assertEqual(claim["proof_grade"]["grade"], "G2")

    def test_label_only_trend_grades_below_its_label(self):
        from nomorals.media.explore.trends import proof_grade_for_claim

        legs = {"sweeps": ["s1"], "instruments": ["web_search"],
                "source_classes": ["web"], "corroborated_urls": 0,
                "velocity": None, "single_source": True}
        labelled = {"kind": "trend", "claim_key": "beat x",
                    "confidence_basis": "1 sweep(s): s1",
                    "basis": legs, "backing": {"observations": []}}
        grade = proof_grade_for_claim(labelled).grade
        self.assertEqual(grade, ProofGrade.G1)
        self.assertLess(grade, grade_for_ws4_kind("trend"))
        # the same label WITH its corroboration legs attached
        legs2 = dict(legs, sweeps=["s1", "s2"])
        backed = dict(labelled, basis=legs2)
        self.assertEqual(proof_grade_for_claim(backed).grade,
                         ProofGrade.G4)
        # and with no basis record at all: G0
        naked = {"kind": "trend", "claim_key": "beat x",
                 "confidence_basis": "", "basis": {}}
        self.assertEqual(proof_grade_for_claim(naked).grade,
                         ProofGrade.G0)

    def test_velocity_never_upgrades_a_grade(self):
        from nomorals.media.explore.trends import proof_grade_for_claim

        storm = {"ratio": 99.0, "rising": True, "latest": 500,
                 "baseline_mean": 1.0, "prior_sweeps": 4,
                 "from_zero": False}
        unv = {"kind": "unverified", "claim_key": "beat y",
               "confidence_basis": "1 sweep(s): s1",
               "basis": {"sweeps": ["s1"],
                         "source_classes": ["video"],
                         "corroborated_urls": 0, "velocity": storm},
               "backing": {"observations": []}}
        self.assertEqual(proof_grade_for_claim(unv).grade,
                         ProofGrade.G1)
        # the same velocity on a trend-labelled row with no
        # corroboration legs still earns nothing past intake
        trd = dict(unv, kind="trend")
        self.assertEqual(proof_grade_for_claim(trd).grade,
                         ProofGrade.G1)
        # but the identical derivation IS the spike's evidence
        spk = dict(unv, kind="spike")
        self.assertEqual(proof_grade_for_claim(spk).grade,
                         ProofGrade.G2)

    def test_stored_rows_carry_the_stamp_and_reads_rederive(self):
        from nomorals.media.explore import ExploreStore, current_claims, evaluate_trends

        store = ExploreStore()
        s1 = _sweep(store, ["gold trading"], at=1000.0)
        _obs(store, s1, "gold trading", "youtube_search",
             _rows(3, "g1"), at=1001.0)
        s2 = _sweep(store, ["gold trading"], at=2000.0)
        _obs(store, s2, "gold trading", "web_search",
             _rows(4, "g3"), at=2001.0)
        evaluate_trends(store, now=2100.0)
        stored = store.claims(limit=10)
        self.assertTrue(stored)
        for row in stored:
            self.assertEqual(row["proof_grade"]["grade"], "G4")
        # a legacy row planted WITHOUT a stamp still grades at
        # read — the derivation governs, not the stored field
        legacy = dict(stored[0])
        legacy.pop("proof_grade")
        legacy["claim_key"] = "legacy planted key"
        legacy["claim_id"] = "claim:legacy-planted-key"
        store.record_claim(legacy)
        current = current_claims(store, now=2100.0)
        planted = _claim_for(current, "gold trading")
        self.assertEqual(planted["proof_grade"]["grade"], "G4")
        # a fiat stamp ABOVE the evidence is not honoured either:
        fiat = dict(legacy)
        fiat["claim_key"] = "fiat planted key"
        fiat["claim_id"] = "claim:fiat-planted-key"
        fiat["subject"] = "fiat subject"
        fiat["kind"] = "trend"
        fiat["basis"] = {"sweeps": ["s1"],
                         "source_classes": ["web"],
                         "corroborated_urls": 0, "velocity": None,
                         "single_source": True}
        fiat["subject_kind"] = "beat"
        fiat["proof_grade"] = {"grade": "G6", "evidence": []}
        store.record_claim(fiat)
        current = current_claims(store, now=2100.0)
        fiat_row = _claim_for(current, "fiat subject")
        self.assertEqual(fiat_row["proof_grade"]["grade"], "G1")

    def test_renders_and_feed_carry_the_grade(self):
        from nomorals.media.explore.board import render_explore_board
        from nomorals.media.explore.feed import claims_as_findings
        from nomorals.media.explore.tool import render_claims_text

        store, report = _trend_store()
        claims = report["claims_current"]
        text = render_claims_text(claims, report)
        self.assertIn("proof G4", text)
        self.assertIn("[trend]", text)
        board = render_explore_board(store=store, now=2100.0)
        self.assertIn("proof G4", board)
        findings = claims_as_findings(store, now=2100.0)
        self.assertTrue(findings)
        self.assertEqual(findings[0]["proof_grade"]["grade"], "G4")


# ── (g) the import route, not a mirror ──────────────────────────


class ImportRouteTests(unittest.TestCase):

    def test_both_surfaces_use_the_core_machinery(self):
        from nomorals.agents.research_digest import claim_proof_grade
        from nomorals.media.explore.trends import proof_grade_for_claim

        digest_grade = claim_proof_grade(_digest_claim(
            "pg3-m", "Postgres outperforms MySQL for analytical "
                     "workloads on large datasets",
            sources=("https://a.example/1",)))
        self.assertIsInstance(digest_grade, ClaimGrade)
        ws4_grade = proof_grade_for_claim({
            "kind": "unverified", "claim_key": "beat z",
            "confidence_basis": "1 sweep(s): s1",
            "basis": {"sweeps": ["s1"], "source_classes": ["web"]},
            "backing": {"observations": []}})
        self.assertIsInstance(ws4_grade, ClaimGrade)


if __name__ == "__main__":
    unittest.main()
