"""Pass M3 — Library-fed trivia: the verified bank behind TriviaForge.

Pins the design's guarantees:
* bank admission is fail-closed — a question whose evidence is not
  in the source passage, or whose answer is absent from the
  evidence, is DISCARDED (counted), never served;
* brain-posed maths is believed only after an independent
  computation reproduces the claimed answer; procedural MathClass
  maths banks with its worked solution;
* deal() mixes the sources — the bank takes >= half a dealt set
  when stocked, the generative forge serves the rest, and an empty
  bank degrades gracefully to the pre-M3 deal;
* served bank questions join the anti-repeat window, and their
  underlying teaching is marked served in the Teachings ledger
  (surface="trivia") so motivation/pulses inherit both cooldowns;
* the refiller maintains stock across ticks and the bank stats are
  honest (candidates == admitted + discarded, reasons sum out).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import nomorals.agents.teachings as teach_mod
import nomorals.games.trivia_forge as TF
from nomorals.agents.teachings import TeachingsLedger
from nomorals.games.trivia_forge import (
    TriviaForge, TriviaLadder, verify_library_candidate,
    verify_llm_maths)
from nomorals.storage.db import Database


def _ctx(test: unittest.TestCase) -> SimpleNamespace:
    tmp = tempfile.mkdtemp(prefix="m3-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        workspace_dir=tmp, resolve=lambda rel: os.path.join(tmp, rel))
    return SimpleNamespace(db=db, settings=settings, extras={})


# ── fixture passages + a scripted brain ──────────────────────────────
# Each passage carries one checkable fact with a unique marker name;
# the scripted suggest answers per-passage by marker, so generation
# stays deterministic however the refiller shuffles its pick order.

_PASSAGES = [
    {"marker": "Zeno", "theme": "stoicism", "ref": "zeno-lives",
     "label": "Lives of the Stoics — WisdomKeeper",
     "passage": ("Zeno of Citium founded the Stoic school in Athens "
                 "around 300 BC, teaching in the painted porch. Zeno "
                 "held that virtue is the only true good, and that "
                 "wealth and health are indifferent things, preferred "
                 "or dispreferred but never good in themselves."),
     "q": "Who founded the Stoic school in Athens?",
     "a": "Zeno of Citium",
     "d": ["Marcus Aurelius", "Epictetus", "Seneca"],
     "ev": "Zeno of Citium founded the Stoic school in Athens"},
    {"marker": "Seneca", "theme": "time", "ref": "seneca-letters",
     "label": "Letters — WisdomKeeper",
     "passage": ("Seneca wrote that we suffer more in imagination "
                 "than in reality, and that time is the one thing "
                 "every person spends without ever auditing the "
                 "account. Seneca urged Lucilius to hold every hour "
                 "in his grasp and depend less on tomorrow."),
     "q": "According to Seneca, where do we suffer more?",
     "a": "in imagination",
     "d": ["in dreams", "in memory", "in exile"],
     "ev": "we suffer more in imagination than in reality"},
    {"marker": "Epictetus", "theme": "control", "ref": "enchiridion",
     "label": "Enchiridion — WisdomKeeper",
     "passage": ("Epictetus taught that some things are up to us and "
                 "some things are not up to us. Epictetus said the "
                 "things up to us are opinion, pursuit, desire and "
                 "aversion — in short, whatever is our own doing, "
                 "while the body and property are not up to us."),
     "q": "In the Enchiridion, what is up to us?",
     "a": "opinion",
     "d": ["the body", "property", "reputation"],
     "ev": "the things up to us are opinion, pursuit, desire and "
           "aversion"},
]


def _lib_item(p: dict) -> dict:
    return {
        "id": f"teach_{p['ref']}", "kind": "teaching",
        "title": p["label"].split(" — ")[0], "body": "",
        "theme": p["theme"], "tradition": "stoicism", "weight": 1.0,
        "source": {"type": "wisdom", "ref": p["ref"],
                   "label": p["label"], "passage": p["passage"]},
        "created_at": 0.0, "served": [],
    }


def _scripted_suggest(*, maths_answer: str = "96",
                      lib_mode: str = "good"):
    """A brain that answers each prompt class deterministically.

    lib_mode: "good" (gate-passing library JSON), "bad_evidence"
    (evidence quote absent from the passage), "bad_answer" (answer
    absent from the evidence).
    """
    def suggest(prompt: str) -> str:
        if "Solve it yourself" in prompt:
            # the semantic gate's independent solve: the scripted
            # brain works the baker story blind and lands on the
            # same answer by the same operation.
            if "12 trays of 8 rolls" in prompt:
                return json.dumps({
                    "answer": "96", "op": "multiply",
                    "why": "multiply: 12 trays with 8 rolls in each"})
            return "{}"
        if "WORD problem" in prompt:
            return json.dumps({
                "question": "A baker makes 12 trays of 8 rolls for "
                            "the morning market. How many rolls does "
                            "the baker make?",
                "answer": maths_answer, "expression": "12 * 8"})
        if "Using ONLY the passage" in prompt:
            for p in _PASSAGES:
                if p["marker"] in prompt:
                    ev = p["ev"]
                    ans = p["a"]
                    if lib_mode == "bad_evidence":
                        ev = "a sentence that was never in the passage"
                    if lib_mode == "bad_answer":
                        ans = "the moon"
                    return json.dumps({
                        "question": p["q"], "answer": ans,
                        "distractors": p["d"], "evidence_quote": ev})
            return "{}"
        if "trivia questions about" in prompt:
            return json.dumps(
                [["What is the capital of France?", "Paris"],
                 ["Which planet is known as the Red Planet?", "Mars"],
                 ["What is the largest ocean on Earth?", "Pacific"],
                 ["Who painted the Mona Lisa?", "Da Vinci"],
                 ["What is the chemical symbol for gold?", "Au"],
                 ["How many sides does a hexagon have?", "Six"],
                 ["What is the fastest land animal?", "Cheetah"],
                 ["Which gas do plants absorb?", "Carbon dioxide"]])
        return ""
    return suggest


def _forge(test: unittest.TestCase, **kw) -> TriviaForge:
    ctx = _ctx(test)
    kw.setdefault("context", ctx)
    kw.setdefault("db", ctx.db)
    return TriviaForge(**kw)


def _no_autotick(f: TriviaForge) -> TriviaForge:
    """Freeze the deal-time top-up so a test sees exactly the stock
    it banked (the autotick is a separate timed mechanism)."""
    import time as _time
    f._last_autotick = _time.time()
    return f


def _patch_passages(test: unittest.TestCase, items=None):
    items = items if items is not None else \
        [_lib_item(p) for p in _PASSAGES]
    patcher = mock.patch.object(
        teach_mod, "library_candidates",
        lambda context, themes: list(items))
    patcher.start()
    test.addCleanup(patcher.stop)
    return items


# ── the gates, directly ──────────────────────────────────────────────

class GateTests(unittest.TestCase):
    def setUp(self):
        self.p = _PASSAGES[0]
        self.good = {"question": self.p["q"], "answer": self.p["a"],
                     "distractors": list(self.p["d"]),
                     "evidence_quote": self.p["ev"]}

    def test_clean_candidate_admitted(self):
        ok, reason = verify_library_candidate(self.p["passage"],
                                              dict(self.good))
        self.assertTrue(ok, reason)

    def test_evidence_not_in_source_discarded(self):
        cand = dict(self.good,
                    evidence_quote="Zeno was born on the moon")
        ok, reason = verify_library_candidate(self.p["passage"], cand)
        self.assertFalse(ok)
        self.assertEqual(reason, "evidence not in source")

    def test_answer_absent_from_evidence_discarded(self):
        cand = dict(self.good, answer="the moon")
        ok, reason = verify_library_candidate(self.p["passage"], cand)
        self.assertFalse(ok)
        self.assertEqual(reason, "answer not in evidence")

    def test_distractor_in_evidence_discarded(self):
        cand = dict(self.good, distractors=[
            "Athens", "Marcus Aurelius", "Seneca"])
        ok, reason = verify_library_candidate(self.p["passage"], cand)
        self.assertFalse(ok)
        self.assertEqual(reason, "distractor appears in evidence")

    def test_llm_maths_wrong_answer_rejected(self):
        ok, reason, _ = verify_llm_maths(
            "A baker makes 12 trays of 8 rolls. How many rolls?",
            "84", "12 * 8")
        self.assertFalse(ok)
        self.assertEqual(reason, "answer mismatch")

    def test_llm_maths_ungrounded_expression_rejected(self):
        ok, reason, _ = verify_llm_maths(
            "Sara has 5 apples and buys 3 more. How many apples?",
            "68", "5 + 3 + 60")
        self.assertFalse(ok)
        self.assertEqual(reason, "expression number not in question")

    def test_llm_maths_right_answer_verified(self):
        ok, reason, computed = verify_llm_maths(
            "A baker makes 12 trays of 8 rolls. How many rolls?",
            "96", "12 * 8")
        self.assertTrue(ok, reason)
        self.assertEqual(computed, 96.0)


# ── the refiller: admission, discards, stats ─────────────────────────

class RefillTests(unittest.TestCase):
    def test_clean_library_question_banked_with_provenance(self):
        _patch_passages(self)
        f = _forge(self, suggest=_scripted_suggest(), seed=3)
        rep = f.refill_bank(max_passages=3, max_maths=0,
                            max_maths_llm=0)
        self.assertEqual(rep["admitted"], 3, rep)
        bank = [e for e in f.bank.items() if e["kind"] == "library"]
        self.assertEqual(len(bank), 3)
        entry = bank[0]
        self.assertEqual(entry["source"]["type"], "wisdom")
        self.assertTrue(entry["teach"]["_key"])
        self.assertEqual(TF.source_tag(entry),
                         f"📚 From her library: "
                         f"{entry['source']['label']}")

    def test_bad_evidence_discarded_never_banked(self):
        _patch_passages(self)
        f = _forge(self, suggest=_scripted_suggest(
            lib_mode="bad_evidence"), seed=3)
        rep = f.refill_bank(max_passages=3, max_maths=0,
                            max_maths_llm=0)
        self.assertEqual(rep["admitted"], 0, rep)
        self.assertEqual(rep["discarded"], 3, rep)
        self.assertEqual(f.bank.stock_count(), 0)
        stats = f.bank_status()
        self.assertEqual(
            stats["discard_reasons"].get("evidence not in source"), 3)

    def test_bad_answer_discarded_never_banked(self):
        _patch_passages(self)
        f = _forge(self, suggest=_scripted_suggest(
            lib_mode="bad_answer"), seed=3)
        rep = f.refill_bank(max_passages=3, max_maths=0,
                            max_maths_llm=0)
        self.assertEqual(rep["admitted"], 0, rep)
        stats = f.bank_status()
        self.assertEqual(
            stats["discard_reasons"].get("answer not in evidence"), 3)

    def test_llm_maths_wrong_then_right(self):
        f = _forge(self, suggest=_scripted_suggest(maths_answer="84"),
                   seed=3)
        rep = f.refill_bank(max_passages=0, max_maths=0,
                            max_maths_llm=1)
        self.assertEqual(rep["admitted"], 0, rep)
        self.assertEqual(f.bank_status()["discard_reasons"].get(
            "answer mismatch"), 1)
        f2 = _forge(self, suggest=_scripted_suggest(maths_answer="96"),
                    seed=3)
        rep2 = f2.refill_bank(max_passages=0, max_maths=0,
                              max_maths_llm=1)
        self.assertEqual(rep2["admitted"], 1, rep2)
        entry = [e for e in f2.bank.items()
                 if e["kind"] == "maths_llm"][0]
        self.assertEqual(entry["answer"], "96")
        self.assertIn("12 * 8", entry["evidence"])

    def test_procedural_maths_banked_with_solution(self):
        f = _forge(self, suggest=None, seed=3)
        rep = f.refill_bank(max_passages=0, max_maths=8,
                            max_maths_llm=0)
        self.assertGreaterEqual(rep["admitted"], 6, rep)
        maths = [e for e in f.bank.items() if e["kind"] == "maths"]
        self.assertTrue(maths)
        for e in maths:
            self.assertTrue(e["solution"])
            self.assertEqual(len(e["distractors"]), 3)
            self.assertNotIn(e["answer"], e["distractors"])

    def test_refiller_maintains_stock_across_ticks(self):
        _patch_passages(self)
        f = _forge(self, suggest=_scripted_suggest(), seed=3)
        stocks = []
        for _ in range(3):
            f.refill_bank()
            stocks.append(f.bank.stock_count())
        self.assertGreater(stocks[1], stocks[0], stocks)
        self.assertGreater(stocks[2], stocks[1], stocks)
        stats = f.bank_status()
        self.assertEqual(stats["ticks"], 3)
        # honesty: every candidate is admitted or discarded
        self.assertEqual(stats["candidates"],
                         stats["admitted"] + stats["discarded"])
        self.assertEqual(sum(stats["discard_reasons"].values()),
                         stats["discarded"])

    def test_tick_at_target_is_counted_noop(self):
        _patch_passages(self)
        f = _forge(self, suggest=_scripted_suggest(), seed=3)
        with mock.patch.object(TF, "BANK_TARGET", 10):
            f.refill_bank()
            self.assertGreaterEqual(f.bank.stock_count(), 10)
            rep = f.refill_bank()
            self.assertEqual(rep["generated"], 0, rep)
        self.assertIn("10", f.render_bank_status())


# ── the deal mix ─────────────────────────────────────────────────────

class DealMixTests(unittest.TestCase):
    def _stocked_forge(self) -> TriviaForge:
        _patch_passages(self)
        f = _forge(self, suggest=_scripted_suggest(), seed=11)
        f.refill_bank(max_passages=3, max_maths=8, max_maths_llm=1)
        self.assertGreaterEqual(f.bank.stock_count(), 10)
        return _no_autotick(f)

    def test_bank_takes_at_least_half_when_stocked(self):
        f = self._stocked_forge()
        recs = f.deal_detailed(count=8)
        self.assertEqual(len(recs), 8)
        bank_kinds = {"library", "maths", "maths_llm"}
        from_bank = [r for r in recs if r["kind"] in bank_kinds]
        self.assertGreaterEqual(len(from_bank), 4, recs)
        # the other source still serves — nothing was retired
        self.assertTrue(any(r["kind"] == "generative" for r in recs),
                        recs)
        lib = [r for r in recs if r["kind"] == "library"]
        for r in lib:
            self.assertTrue(r["tag"].startswith("📚 From her library:"))

    def test_graceful_generative_only_when_bank_empty(self):
        f = _no_autotick(_forge(self, suggest=_scripted_suggest(),
                                seed=11))
        recs = f.deal_detailed(count=4)
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(r["kind"] == "generative" for r in recs))

    def test_graceful_template_only_without_brain_or_bank(self):
        f = TriviaForge(seed=11)
        qs = f.deal(count=6)
        self.assertEqual(len(qs), 6)
        self.assertEqual(len({q for q, _ in qs}), 6)

    def test_served_bank_question_not_redealt_inside_window(self):
        f = self._stocked_forge()
        first = f.deal_detailed(count=8)
        bank_qs = {r["q"] for r in first
                   if r["kind"] in ("library", "maths", "maths_llm")}
        self.assertTrue(bank_qs)
        second = f.deal_detailed(count=8)
        again = {r["q"] for r in second} & bank_qs
        self.assertFalse(again, again)


# ── cross-surface: the Teachings ledger binding ─────────────────────

class LedgerBindingTests(unittest.TestCase):
    def test_trivia_serve_marks_teaching_and_cools_theme(self):
        items = _patch_passages(self, [_lib_item(_PASSAGES[0])])
        ctx_forge = _forge(self, suggest=_scripted_suggest(), seed=5)
        ctx = ctx_forge._context
        ctx_forge.refill_bank(max_passages=1, max_maths=0,
                              max_maths_llm=0)
        _no_autotick(ctx_forge)
        # one library entry banked; force a deal that must serve it:
        # drain everything else by dealing the bank dry first is
        # overkill — a 1-count deal takes the bank's share (ceil .5).
        recs = ctx_forge.deal_detailed(count=1)
        self.assertEqual(recs[0]["kind"], "library", recs)
        ledger = TeachingsLedger(ctx)
        served = ledger.served_items(surface_prefix="trivia")
        self.assertEqual(len(served), 1, served)
        self.assertEqual(served[0]["theme"], "stoicism")
        # the theme is now in the 48h cross-surface cooldown and the
        # item itself is invisible to every surface for 30 days
        self.assertTrue(ledger.theme_in_cooldown("stoicism"))
        self.assertIsNone(ledger.pick(surface="motivation",
                                       theme="stoicism"))
        ledger._store_candidates(items, __import__("time").time())
        pool = ledger.candidates()
        self.assertFalse(
            [it for it in pool
             if (it.get("source") or {}).get("ref") == "zeno-lives"],
            pool)


# ── provenance in play: the ladder ───────────────────────────────────

class LadderProvenanceTests(unittest.TestCase):
    def test_tag_after_answer_not_on_card(self):
        tag = "📚 From her library: Lives of the Stoics — WisdomKeeper"
        lad = TriviaLadder(
            [("Who founded the Stoic school?", "Zeno",
              ["Zeno", "Seneca", "Plato", "Epicurus"], tag, "")],
            seed=0)
        self.assertNotIn("From her library", lad.ask())
        over, msg = lad.answer("Zeno")
        self.assertIn(tag, msg)

    def test_solution_teaches_on_a_miss(self):
        lad = TriviaLadder(
            [("What is 12 × 8?", "96", ["96", "84", "106", "69"],
              "", "12 × 8 = 96")], seed=0)
        over, msg = lad.answer("84")
        self.assertTrue(over)
        self.assertIn("🧮 12 × 8 = 96", msg)

    def test_deal_ladder_carries_tags(self):
        _patch_passages(self)
        f = _forge(self, suggest=_scripted_suggest(), seed=5)
        f.refill_bank(max_passages=3, max_maths=0, max_maths_llm=0)
        _no_autotick(f)
        rows = f.deal_ladder(count=6)
        self.assertEqual(len(rows), 6)
        for q, a, opts, tag, sol in rows:
            self.assertEqual(len(opts), 4)
            self.assertIn(a, opts)
        self.assertTrue(any(r[3].startswith("📚") for r in rows))


# ── the wisdom-tick stage ────────────────────────────────────────────

class WisdomTickStageTests(unittest.TestCase):
    def test_wisdom_tick_maintains_the_bank(self):
        from nomorals.wisdom.autonomy import WisdomOrgan

        ctx = _ctx(self)
        organ = WisdomOrgan(ctx)
        # the corpus stages are not this test's subject (and would
        # reach for the live archive manifest); the trivia stage is.
        # No brain in this test: the stage must still bank maths.
        with mock.patch("nomorals.wisdom.autonomy._brain_suggest",
                        return_value=None), \
                mock.patch.object(WisdomOrgan, "_drain_events",
                                  return_value=0), \
                mock.patch.object(WisdomOrgan, "_ingest_pending",
                                  return_value=None), \
                mock.patch.object(WisdomOrgan, "_digest_new",
                                  return_value=None), \
                mock.patch.object(WisdomOrgan, "_cross_link",
                                  return_value=0):
            rep = organ.tick(budget_seconds=120)
        self.assertIn("trivia_bank", rep, rep.get("errors"))
        self.assertNotIn("trivia_bank",
                         " ".join(rep.get("errors", [])), rep["errors"])
        stage = rep["trivia_bank"]
        self.assertGreater(stage["stock"], 0, stage)
        # the organ's bank is the same store the forge reads
        f = TriviaForge(db=ctx.db, context=ctx)
        self.assertEqual(f.bank.stock_count(), stage["stock"])


if __name__ == "__main__":
    unittest.main()
