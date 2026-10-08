"""Offline tests for build-map #99 — brief-first content pipeline."""

import os
import tempfile
import unittest

from nomorals.marketing.briefs import (
    BriefStore, ContentBrief, control_brief, control_content,
)


def _db():
    return os.path.join(tempfile.mkdtemp(), "briefs.db")


MOCK_RESULTS = [
    {"title": "Lagos rents explained",
     "snippet": "How much is rent in Lagos? 2 bedroom flats in Yaba cost around 2m per year. What should you look for when renting? Always inspect before paying."},
    {"title": "Yaba rental guide",
     "snippet": "Why are Yaba rents rising? The area is close to tech hubs. How do you negotiate with Lagos landlords? Get everything in writing."},
]


def _mock_search(topic):
    return MOCK_RESULTS


class BriefBuildingTests(unittest.TestCase):
    def test_build_brief_returns_brief(self):
        s = BriefStore(db_path=_db())
        b = s.build_brief("Lagos rents", search_fn=_mock_search)
        self.assertIsNotNone(b)
        self.assertTrue(b.brief_id)
        self.assertEqual(b.topic, "Lagos rents")

    def test_brief_has_keywords_questions_angles_outline(self):
        s = BriefStore(db_path=_db())
        b = s.build_brief("Lagos rents", search_fn=_mock_search)
        self.assertGreater(len(b.keywords), 0)
        self.assertGreater(len(b.questions), 0)
        self.assertGreater(len(b.angles), 0)
        self.assertGreater(len(b.outline), 0)

    def test_brief_topic_words_seeded(self):
        s = BriefStore(db_path=_db())
        b = s.build_brief("Lagos rents", search_fn=_mock_search)
        kw = " ".join(b.keywords).lower()
        self.assertIn("lagos", kw)

    def test_build_brief_no_search_still_works(self):
        s = BriefStore(db_path=_db())
        b = s.build_brief("Lagos rents")
        self.assertIsNotNone(b)
        self.assertGreater(len(b.questions), 0)  # heuristic fallbacks

    def test_build_brief_empty_topic(self):
        s = BriefStore(db_path=_db())
        self.assertIsNone(s.build_brief(""))

    def test_build_brief_search_raises(self):
        s = BriefStore(db_path=_db())
        def boom(t):
            raise RuntimeError("nope")
        b = s.build_brief("Lagos rents", search_fn=boom)
        self.assertIsNotNone(b)

    def test_brief_persisted_and_retrievable(self):
        s = BriefStore(db_path=_db())
        b = s.build_brief("Lagos rents", search_fn=_mock_search)
        got = s.get_brief(b.brief_id)
        self.assertIsNotNone(got)
        self.assertEqual(got.topic, "Lagos rents")
        self.assertEqual(got.keywords, b.keywords)

    def test_latest_brief(self):
        s = BriefStore(db_path=_db())
        s.build_brief("Lagos rents", search_fn=_mock_search)
        got = s.latest_brief("rents")
        self.assertIsNotNone(got)


class ScoringTests(unittest.TestCase):
    def _brief(self, s):
        return s.build_brief("Lagos rents", search_fn=_mock_search)

    def test_score_draft_returns_score(self):
        s = BriefStore(db_path=_db())
        b = self._brief(s)
        sc = s.score_draft("Lagos rents are rising in Yaba this year", b)
        self.assertGreaterEqual(sc.score, 0)
        self.assertLessEqual(sc.score, 100)

    def test_term_coverage_rewards_keywords(self):
        s = BriefStore(db_path=_db())
        b = self._brief(s)
        rich = " ".join(b.keywords[:8]) + " — Lagos rents explained."
        poor = "unrelated words about something else entirely here."
        sc_rich = s.score_draft(rich, b)
        sc_poor = s.score_draft(poor, b)
        self.assertGreater(sc_rich.term_coverage, sc_poor.term_coverage)

    def test_missing_terms_and_suggestions(self):
        s = BriefStore(db_path=_db())
        b = self._brief(s)
        sc = s.score_draft("short text", b)
        self.assertGreater(len(sc.missing_terms), 0)
        self.assertGreater(len(sc.suggestions), 0)

    def test_score_empty_draft(self):
        s = BriefStore(db_path=_db())
        b = self._brief(s)
        sc = s.score_draft("", b)
        self.assertEqual(sc.score, 0.0)

    def test_score_never_raises_on_garbage(self):
        s = BriefStore(db_path=_db())
        sc = s.score_draft(None, None)
        self.assertIsInstance(sc.score, float)


class EnforcementTests(unittest.TestCase):
    def test_schedule_refuses_no_brief(self):
        s = BriefStore(db_path=_db())
        # Craft a draft row with no brief attached.
        import time as _t
        s._save_draft(type("D", (), {
            "draft_id": "bd_nobrief", "topic": "x", "content": "hello",
            "brief_id": "", "platform": "x", "score": 50.0,
            "predicted_engagement": 50.0, "scheduled_for": 0.0,
            "published_at": 0.0, "outcome_metrics": {}, "created_at": _t.time(),
        })())
        res = s.schedule_draft("bd_nobrief", _t.time() + 3600)
        self.assertFalse(res["ok"])
        self.assertIn("brief", res["reason"].lower())

    def test_schedule_ok_with_brief(self):
        s = BriefStore(db_path=_db())
        b = s.build_brief("Lagos rents", search_fn=_mock_search)
        d = s.draft_from_brief(b)
        self.assertIsNotNone(d)
        res = s.schedule_draft(d.draft_id, 9999999999.0)
        self.assertTrue(res["ok"])

    def test_schedule_unknown_draft(self):
        s = BriefStore(db_path=_db())
        res = s.schedule_draft("bd_missing", 9999999999.0)
        self.assertFalse(res["ok"])

    def test_draft_requires_brief(self):
        s = BriefStore(db_path=_db())
        self.assertIsNone(s.draft_from_brief(None))


class PredictionFeedbackTests(unittest.TestCase):
    def test_predict_performance_band(self):
        s = BriefStore(db_path=_db())
        b = s.build_brief("Lagos rents", search_fn=_mock_search)
        p = s.predict_performance("Why are Lagos rents rising? Here's the take.", b)
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 100)

    def test_record_outcome_feeds_back(self):
        s = BriefStore(db_path=_db())
        b = s.build_brief("Lagos rents", search_fn=_mock_search)
        d = s.draft_from_brief(b)
        # Feed 5 outcomes so calibration activates.
        for i in range(5):
            s.record_outcome(d.draft_id, {"engagement_score": 80.0})
        sc = s.score_draft(d.content, b)
        self.assertEqual(sc.source, "learned")
        self.assertTrue(s.record_outcome(d.draft_id, {"engagement_score": 60}))

    def test_record_outcome_unknown_draft(self):
        s = BriefStore(db_path=_db())
        self.assertFalse(s.record_outcome("bd_nope", {}))


class PipelineTests(unittest.TestCase):
    def test_full_run(self):
        s = BriefStore(db_path=_db())
        res = s.run("Lagos rents", search_fn=_mock_search)
        self.assertTrue(res["ok"])
        self.assertIsNotNone(res["brief"])
        self.assertIsNotNone(res["draft"])
        self.assertGreater(res["score"].score, 0)
        self.assertGreater(res["predicted_engagement"], 0)

    def test_run_with_schedule(self):
        s = BriefStore(db_path=_db())
        res = s.run("Lagos rents", search_fn=_mock_search,
                    schedule_at=9999999999.0)
        self.assertTrue(res["ok"])
        self.assertTrue(res["schedule"]["ok"])

    def test_run_empty_topic(self):
        s = BriefStore(db_path=_db())
        res = s.run("")
        self.assertFalse(res["ok"])

    def test_draft_content_mentions_brief_terms(self):
        s = BriefStore(db_path=_db())
        res = s.run("Lagos rents", search_fn=_mock_search)
        content = res["draft"].content.lower()
        self.assertTrue(any(k in content for k in res["brief"].keywords[:6]))


class ChatTests(unittest.TestCase):
    def test_brief_help(self):
        self.assertIn("/brief", control_brief("", store=BriefStore(db_path=_db())))

    def test_brief_builds(self):
        s = BriefStore(db_path=_db())
        out = control_brief("run Lagos rents", store=s)
        self.assertIn("Brief", out)

    def test_content_score_needs_brief(self):
        s = BriefStore(db_path=_db())
        out = control_content("score hello world", store=s)
        self.assertIn("brief", out.lower())

    def test_content_score_with_brief(self):
        s = BriefStore(db_path=_db())
        s.build_brief("Lagos rents", search_fn=_mock_search)
        out = control_content("score Lagos rents are wild right now", store=s)
        self.assertIn("draft score", out)

    def test_content_schedule_refuses(self):
        s = BriefStore(db_path=_db())
        out = control_content("schedule bd_nope 9999999999", store=s)
        self.assertIn("❌", out)

    def test_content_help(self):
        self.assertIn("/content", control_content("", store=BriefStore(db_path=_db())))

    def test_chat_never_raises(self):
        s = BriefStore(db_path=_db())
        for fn, tail in [(control_brief, None), (control_content, None),
                         (control_brief, "x" * 500), (control_content, "score "),
                         (control_content, "schedule only-one-arg")]:
            self.assertIsInstance(fn(tail, store=s), str)


if __name__ == "__main__":
    unittest.main()
