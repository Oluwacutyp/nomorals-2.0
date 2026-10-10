"""Tests for gig application drafting pipeline (no network, no LLM)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents import opportunities as O
from nomorals.agents import gig_applier as G


def _opp(**kw):
    base = dict(kind="paid_task", title="AI training gig",
                source="outlier.ai", url="https://outlier.ai/apply",
                payout_text="$20/hr", effort="medium",
                regions=["global"], skills=["ai training"])
    base.update(kw)
    return O.Opportunity(**base)


def _settings():
    tmp = tempfile.mkdtemp()
    return SimpleNamespace(resolve=lambda p: str(Path(tmp) / p))


class MatchScoreTests(unittest.TestCase):
    def test_full_match(self):
        p = O.OpportunityProfile(skills=["ai training"], regions=["global"])
        self.assertEqual(1.0, O.match_score(_opp(), p))

    def test_no_skill_match(self):
        p = O.OpportunityProfile(skills=["welding"], regions=["global"])
        self.assertLess(O.match_score(_opp(), p), 1.0)

    def test_region_mismatch(self):
        p = O.OpportunityProfile(skills=[], regions=["nigeria"])
        self.assertEqual(0.0, O.match_score(_opp(), p))

    def test_empty_profile(self):
        p = O.OpportunityProfile(skills=[], regions=[])
        self.assertEqual(1.0, O.match_score(_opp(), p))


class BoardFinderTests(unittest.TestCase):
    def test_finders_registered(self):
        names = [f.name for f in O.FINDERS]
        for expected in ("outlier", "mindrift", "upwork", "jobberman"):
            self.assertIn(expected, names)

    def test_outlier_curated(self):
        opps = O.OutlierFinder().curated_opportunities()
        self.assertTrue(opps)
        self.assertTrue(all("outlier" in o.url for o in opps))


class ExplicitSubmitTests(unittest.TestCase):
    def test_submit_words(self):
        for t in ("submit", "apply now", "send it", "go ahead do it"):
            self.assertTrue(G.is_explicit_submit(t), t)

    def test_not_submit(self):
        for t in ("", "maybe later", "draft it", "what is this"):
            self.assertFalse(G.is_explicit_submit(t), t)


class ApplierTests(unittest.TestCase):
    def _applier(self):
        return G.GigApplier(settings=_settings(),
                            llm_fn=lambda p: "Dear team, I am a great fit.")

    def test_draft_and_submit(self):
        applier = self._applier()
        opp = _opp()
        app = applier.draft(opp)
        self.assertEqual("drafted", app.status)
        self.assertIn("great fit", app.draft_text)
        # no board submitter registered → honest "submit_attempted",
        # never a fake "submitted".
        app2 = applier.submit(app.gig_id, explicit=True)
        self.assertEqual("submit_attempted", app2.status)
        self.assertIsNone(app2.submitted_at)
        self.assertIn("next_steps", app2.submission_evidence)

    def test_submit_with_board_submitter(self):
        applier = self._applier()
        saved = dict(G.SUBMITTERS)
        G.register_submitter(
            "outlier.ai",
            lambda app, draft: {"ok": True, "method": "test",
                                "evidence": "confirmation id TEST-1"})
        try:
            app = applier.draft(_opp())
            app2 = applier.submit(app.gig_id, explicit=True)
            self.assertEqual("submitted", app2.status)
            self.assertIsNotNone(app2.submitted_at)
            self.assertIsNotNone(app2.follow_up_at)
        finally:
            G.SUBMITTERS.clear()
            G.SUBMITTERS.update(saved)

    def test_status_pipeline(self):
        applier = self._applier()
        saved = dict(G.SUBMITTERS)
        G.register_submitter(
            "outlier.ai",
            lambda app, draft: {"ok": True, "method": "test",
                                "evidence": "confirmation id TEST-2"})
        try:
            app = applier.draft(_opp())
            applier.submit(app.gig_id, explicit=True)
            applier.set_status(app.gig_id, "interview", notes="screening call")
        finally:
            G.SUBMITTERS.clear()
            G.SUBMITTERS.update(saved)
        got = applier.store.get(app.gig_id)
        self.assertEqual("interview", got.status)
        self.assertEqual("screening call", got.notes)

    def test_invalid_transition_raises(self):
        applier = self._applier()
        app = applier.draft(_opp())
        # drafted → interview skips the pipeline: rejected.
        with self.assertRaises(ValueError):
            applier.set_status(app.gig_id, "interview")

    def test_bad_status_raises(self):
        applier = self._applier()
        app = applier.draft(_opp())
        with self.assertRaises(ValueError):
            applier.set_status(app.gig_id, "bogus")

    def test_submit_missing_raises(self):
        applier = self._applier()
        with self.assertRaises(KeyError):
            applier.submit("nonexistent", explicit=True)


if __name__ == "__main__":
    unittest.main()
