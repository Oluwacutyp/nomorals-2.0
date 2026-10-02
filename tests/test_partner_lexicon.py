"""Tests for the partner's dynamic voice feed (nomorals/partner/lexicon_feed.py)
and its wiring into the responder path.

Covers: blending when terms exist, clean visible fallback when the store
is empty, no crash when the DB is missing, owner override winning, the
seed going through the acquire->score->version pipeline, and fallback-path
blending.
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from nomorals.agents.research_lexicon import LexiconStore
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse
from nomorals.partner.lexicon_feed import (
    LEXICON_MODULE,
    LexiconFeed,
    seed_partner_lexicon,
)
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import SpeechProfile, default_persona
from nomorals.partner.relationship import Relationship
from nomorals.partner.responder import PartnerResponder
from nomorals.storage.db import Database


class FakeRouter:
    """Scripted stand-in LLM router."""

    def __init__(self, replies: list[str] | None = None, *, fail: bool = False) -> None:
        self.replies = list(replies or [])
        self.fail = fail
        self.calls: list[list] = []

    def chat(self, messages, params=None, **kw):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        text = self.replies.pop(0) if self.replies else "mhm, okay"
        return LLMResponse(text=text, model="fake")


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _responder(router, db, *, lexicon=None, rng_seed=0) -> PartnerResponder:
    persona = default_persona()
    mood = MoodEngine(persona.baselines, now=1_000_000.0)
    feed = LexiconFeed(db) if lexicon == "feed" else lexicon
    return PartnerResponder(
        router, persona, mood, Relationship(), None, None,
        rng=random.Random(rng_seed), lexicon=feed,
    )


class LexiconFeedTest(unittest.TestCase):
    def test_empty_db_no_terms(self) -> None:
        feed = LexiconFeed(_db())
        self.assertTrue(feed.available)
        self.assertEqual(feed.terms("catchphrase"), ())
        self.assertFalse(feed.has("catchphrase"))
        note, used, fallback = feed.voice_note(default_persona(), "calm")
        # Nothing dynamic anywhere -> no note, zero influence, and the
        # fallback list says exactly which categories fell back.
        self.assertEqual(note, "")
        self.assertEqual(used, 0)
        self.assertIn("catchphrase", fallback)

    def test_none_db_never_raises(self) -> None:
        feed = LexiconFeed(None)
        self.assertFalse(feed.available)
        self.assertEqual(feed.terms("catchphrase"), ())
        self.assertFalse(feed.has("catchphrase"))
        status = feed.status()
        self.assertFalse(status["available"])
        note, used, fallback = feed.voice_note(default_persona(), "calm")
        self.assertEqual(note, "")
        self.assertEqual(used, 0)

    def test_seed_goes_through_pipeline(self) -> None:
        db = _db()
        result = seed_partner_lexicon(db)
        self.assertTrue(result["seeded"])
        self.assertGreater(len(result["added"]), 0)
        store = LexiconStore(SimpleNamespace(db=db))
        self.assertGreater(store.version(LEXICON_MODULE), 0)
        # Every seeded term is scored and versioned like a research term.
        for row in store.terms_for(LEXICON_MODULE, "catchphrase", limit=50):
            self.assertGreaterEqual(row["score"], 0.35)
            self.assertGreaterEqual(row["version"], 1)

    def test_seed_idempotent(self) -> None:
        db = _db()
        first = seed_partner_lexicon(db)
        second = seed_partner_lexicon(db)
        self.assertFalse(second["seeded"])
        self.assertEqual(second["reason"], "already versioned")
        store = LexiconStore(SimpleNamespace(db=db))
        stats = store.stats(LEXICON_MODULE)
        self.assertEqual(stats["total"], len(first["added"]))

    def test_seed_never_readds_retired(self) -> None:
        db = _db()
        # Owner retires a term before any version exists (fresh deploy
        # where the term came from somewhere else): the seed must not
        # resurrect it.
        db.execute(
            "INSERT INTO lexicon_terms (id, term, category, module, score, source, "
            "version, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'retired', ?)",
            ("lex_test1", "cutie", "pet_name", LEXICON_MODULE, 0.5, "owner", 1, 0.0),
        )
        seed_partner_lexicon(db)
        feed = LexiconFeed(db)
        self.assertNotIn("cutie", feed.terms("pet_name", limit=50))

    def test_status_reports_counts(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        status = LexiconFeed(db).status()
        self.assertTrue(status["available"])
        self.assertGreater(status["version"], 0)
        self.assertGreater(status["categories"].get("catchphrase", 0), 0)

    def test_voice_note_blends_static_and_dynamic(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        note, used, _fallback = LexiconFeed(db).voice_note(default_persona(), "calm")
        self.assertGreater(used, 0)
        # Dynamic terms present...
        self.assertIn("okay wait", note)
        # ...alongside the owner's own configured banks (never replaced).
        self.assertIn("ok real", note)  # static catchphrase
        self.assertIn("babe", note)  # static pet name

    def test_voice_note_owner_banks_preserved(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        persona = default_persona()
        persona.speech = SpeechProfile(catchphrases=("my own thing",), pet_names=("skippy",))
        note, _used, _fallback = LexiconFeed(db).voice_note(persona, "calm")
        self.assertIn("my own thing", note)
        self.assertIn("skippy", note)


class ResponderLexiconTest(unittest.TestCase):
    def _system_prompt(self, router) -> str:
        return router.calls[0][0].content

    def test_dynamic_terms_in_prompt(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        router = FakeRouter()
        responder = _responder(router, db, lexicon="feed")
        bundle = responder.respond(chat_platform="local", user_text="hey")
        prompt = self._system_prompt(router)
        self.assertIn("okay wait", prompt)
        self.assertTrue(bundle.lexicon_dynamic)
        self.assertGreater(bundle.lexicon_terms_used, 0)
        d = bundle.to_dict()
        self.assertTrue(d["lexicon_dynamic"])
        self.assertGreater(d["lexicon_terms_used"], 0)

    def test_empty_store_clean_fallback(self) -> None:
        db = _db()  # migrated but never seeded: the honest empty state
        router = FakeRouter()
        responder = _responder(router, db, lexicon="feed")
        with self.assertLogs("nomorals.partner.responder", level="DEBUG") as logs:
            bundle = responder.respond(chat_platform="local", user_text="hey")
        # The reply still works on the static banks...
        self.assertEqual(bundle.parts, ["mhm, okay"])
        # ...but nothing pretends to be dynamic.
        self.assertFalse(bundle.lexicon_dynamic)
        self.assertEqual(bundle.lexicon_terms_used, 0)
        self.assertTrue(any("static bank in use" in m for m in logs.output))

    def test_db_none_no_crash(self) -> None:
        router = FakeRouter()
        responder = _responder(router, None, lexicon=LexiconFeed(None))
        bundle = responder.respond(chat_platform="local", user_text="hey")
        self.assertEqual(bundle.parts, ["mhm, okay"])
        self.assertFalse(bundle.lexicon_dynamic)
        self.assertEqual(bundle.lexicon_terms_used, 0)

    def test_owner_override_disables(self) -> None:
        # lexicon_voice=False in settings -> brain passes lexicon=None.
        # Even with a full store, no dynamic term may influence the reply,
        # while the owner's own catchphrases still show up.
        db = _db()
        seed_partner_lexicon(db)
        router = FakeRouter()
        responder = _responder(router, db, lexicon=None)
        bundle = responder.respond(chat_platform="local", user_text="hey")
        prompt = self._system_prompt(router)
        self.assertNotIn("okay wait", prompt)
        self.assertIn("ok real", prompt)  # owner's static bank intact
        self.assertFalse(bundle.lexicon_dynamic)
        self.assertEqual(bundle.lexicon_terms_used, 0)

    def test_fallback_path_blends(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        dynamic = set(LexiconFeed(db).terms("fallback", limit=12))
        self.assertTrue(dynamic)
        router = FakeRouter(fail=True)
        responder = _responder(router, db, lexicon="feed", rng_seed=7)
        seen_dynamic = 0
        for _ in range(60):
            parts, used = responder._fallback_parts("calm")
            if parts[0] in dynamic:
                seen_dynamic += used
        # Weighted pool -> a dynamic line must surface over 60 draws.
        self.assertGreater(seen_dynamic, 0)

    def test_fallback_bundle_reports_lexicon(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        router = FakeRouter(fail=True)
        responder = _responder(router, db, lexicon="feed", rng_seed=7)
        bundle = responder.respond(chat_platform="local", user_text="hey")
        self.assertTrue(bundle.fallback)
        # The prompt voice note still blended dynamic terms before the
        # provider failed, so the bundle says so.
        self.assertTrue(bundle.lexicon_dynamic)
        self.assertGreater(bundle.lexicon_terms_used, 0)

    def test_guard_extra_phrases(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        # The lexicon's robotic_phrase seed ("i'm here to help") must arm
        # strip_robotic's extra_phrases hook like a hardcoded phrase would.
        router = FakeRouter(replies=["I'm here to help. Anyway hi"])
        responder = _responder(router, db, lexicon="feed")
        bundle = responder.respond(chat_platform="local", user_text="hey")
        self.assertNotIn("help", bundle.text.lower())
        self.assertIn("Anyway hi", bundle.text)


class BrainWiringTest(unittest.TestCase):
    def _context(self, db, **overrides):
        settings = load_settings()
        for key, value in overrides.items():
            setattr(settings.partner, key, value)
        return SimpleNamespace(db=db, router=FakeRouter(), memory=None, settings=settings)

    def test_brain_builds_feed_and_seeds(self) -> None:
        from nomorals.agents.partner_runtime import PartnerBrain

        db = _db()
        brain = PartnerBrain(self._context(db))
        self.assertIsNotNone(brain.lexicon)
        self.assertIs(brain.responder.lexicon, brain.lexicon)
        # Seeded on boot through the pipeline.
        self.assertTrue(brain.lexicon.has("catchphrase"))
        status = brain.lexicon.status()
        self.assertGreater(status["version"], 0)

    def test_brain_lexicon_disabled(self) -> None:
        from nomorals.agents.partner_runtime import PartnerBrain

        db = _db()
        brain = PartnerBrain(self._context(db, lexicon_voice=False, lexicon_seed=False))
        self.assertIsNone(brain.lexicon)
        self.assertIsNone(brain.responder.lexicon)
        # And with the voice off, nothing was seeded either.
        store = LexiconStore(SimpleNamespace(db=db))
        self.assertEqual(store.version(LEXICON_MODULE), 0)

    def test_runtime_status_includes_lexicon(self) -> None:
        from nomorals.agents.partner_runtime import PartnerBrain

        db = _db()
        brain = PartnerBrain(self._context(db))
        # PartnerRuntime.status reads brain.lexicon; emulate the same line.
        lexicon_status = (
            brain.lexicon.status()
            if brain.lexicon is not None
            else {"available": False, "enabled": False, "module": "partner"}
        )
        self.assertTrue(lexicon_status["available"])
        self.assertIn("catchphrase", lexicon_status["categories"])


if __name__ == "__main__":
    unittest.main()
