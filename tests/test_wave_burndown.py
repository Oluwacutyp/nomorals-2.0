"""Wave burn-down — the code-side weaknesses of M3, the weekly hunt
and Pass 8, fixed and pinned:

* **Maths semantic gate** — LLM-posed word problems were admitted on
  arithmetic alone (the expression computes the claimed answer from
  the question's numbers).  A story needing subtraction posed with an
  addition expression passed.  Admission now also requires an
  independent blind solve (question text only) reproducing the
  answer with the same operation family, and the poser's worked line
  is held to its expression's root operation.
* **Trivia purge off the public tool** — ``trivia_bank purge`` rode
  the public games tool (capability social.play, granted to every
  gated chat).  It now refuses by name on that surface; the purge
  lives in ``trivia_bank_purge``, registered with the owner
  capability, and both directions are pinned through a REAL registry.
* **Hunt payout-doc vetting** — the vetter read the candidate page
  only.  A candidate silent on payouts now gets its same-domain
  payout docs (payments / fees / how-it-works) checked before it is
  dropped, the stored evidence names the page that carried the
  proof, and every vetting fetch counts against the run budget.
* **Bet desk session books** — inbox imports were transient report
  rows.  The session-book collector files every import outcome in a
  persistent book (session_books.jsonl), makes re-imports true
  no-ops by content fingerprint, records a corrupt export once by
  name, and surfaces the book in desk status.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.games.trivia_forge import (
    TriviaForge, verify_llm_maths, verify_llm_maths_semantic)
from nomorals.storage.db import Database


# ── shared trivia fixtures ───────────────────────────────────────────

def _ctx(test) -> SimpleNamespace:
    tmp = tempfile.mkdtemp(prefix="wave-")
    if isinstance(test, unittest.TestCase):
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    else:  # pytest function: leak a tmp dir is fine, it is tiny
        pass
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        workspace_dir=tmp, resolve=lambda rel: os.path.join(tmp, rel))
    return SimpleNamespace(db=db, settings=settings, extras={})


_BAKER_Q = ("A baker makes 12 trays of 8 rolls for the morning "
            "market. How many rolls does the baker make?")
_STICKER_Q = ("Lena had 12 stickers. She gave 4 to Ana and 3 to Ben. "
              "How many stickers does Lena have left?")


def _maths_brain(poser: dict, solver: dict):
    """A brain with two desks: the poser (answers WORD-problem
    prompts) and the independent solver (answers the blind
    solve prompt — it sees only the question text)."""
    def suggest(prompt: str) -> str:
        if "Solve it yourself" in prompt:
            return json.dumps(solver)
        if "WORD problem" in prompt:
            return json.dumps(poser)
        return ""
    return suggest


_BAKER_POSER = {"question": _BAKER_Q, "answer": "96",
                "expression": "12 * 8",
                "worked": "multiply the trays by the rolls in each: "
                          "12 * 8"}
_BAKER_SOLVER = {"answer": "96", "op": "multiply",
                 "why": "multiply: 12 trays with 8 rolls in each"}


# ── 1. the semantic gate ─────────────────────────────────────────────

class SemanticGateTests(unittest.TestCase):
    def test_mismatch_passes_the_old_gate(self):
        # the hole being closed: right numbers, right total for the
        # WRONG operation — arithmetic alone cannot see it.
        ok, reason, computed = verify_llm_maths(
            _STICKER_Q, "19", "12 + 4 + 3")
        self.assertTrue(ok, reason)
        self.assertEqual(computed, 19.0)

    def test_mismatch_rejected_by_independent_solve(self):
        brain = _maths_brain(
            {"question": _STICKER_Q, "answer": "19",
             "expression": "12 + 4 + 3",
             "worked": "add all the numbers: 12 + 4 + 3"},
            {"answer": "5", "op": "subtract",
             "why": "subtract the given-away stickers from the 12"})
        ok, reason, _ = verify_llm_maths_semantic(
            _STICKER_Q, "19", "12 + 4 + 3",
            "add all the numbers: 12 + 4 + 3", brain)
        self.assertFalse(ok)
        self.assertIn("independent solve disagrees", reason)

    def test_family_mismatch_rejected_even_when_answer_agrees(self):
        brain = _maths_brain(
            _BAKER_POSER,
            {"answer": "96", "op": "add",
             "why": "add the trays and the rolls together"})
        ok, reason, _ = verify_llm_maths_semantic(
            _BAKER_Q, "96", "12 * 8", _BAKER_POSER["worked"], brain)
        self.assertFalse(ok)
        self.assertIn("independent solve names add", reason)
        self.assertIn("expression computes multiply", reason)

    def test_worked_line_inconsistent_with_expression_rejected(self):
        brain = _maths_brain(
            _BAKER_POSER, _BAKER_SOLVER)
        ok, reason, _ = verify_llm_maths_semantic(
            _BAKER_Q, "96", "12 * 8",
            "subtract the small number from the big one: 12 - 8",
            brain)
        self.assertFalse(ok)
        self.assertIn("worked line inconsistent", reason)

    def test_correct_item_admitted(self):
        brain = _maths_brain(_BAKER_POSER, _BAKER_SOLVER)
        ok, reason, computed = verify_llm_maths_semantic(
            _BAKER_Q, "96", "12 * 8", _BAKER_POSER["worked"], brain)
        self.assertTrue(ok, reason)
        self.assertEqual(computed, 96.0)

    def test_compare_agrees_with_subtraction_root(self):
        brain = _maths_brain(
            {"question": "Ada has 15 pens and Bola has 9. How many "
                         "more pens does Ada have?",
             "answer": "6", "expression": "15 - 9",
             "worked": "subtract: 15 - 9"},
            {"answer": "6", "op": "compare",
             "why": "compare the two counts to find how many more"})
        ok, reason, _ = verify_llm_maths_semantic(
            "Ada has 15 pens and Bola has 9. How many more pens does "
            "Ada have?", "6", "15 - 9", "subtract: 15 - 9", brain)
        self.assertTrue(ok, reason)

    def test_no_brain_no_llm_maths(self):
        ok, reason, _ = verify_llm_maths_semantic(
            _BAKER_Q, "96", "12 * 8", "", None)
        self.assertFalse(ok)
        self.assertIn("no brain for independent solve", reason)

    def test_unparseable_solve_rejected(self):
        brain = _maths_brain(_BAKER_POSER, {})
        # the scripted solver returns "{}" -> no answer to parse
        ok, reason, _ = verify_llm_maths_semantic(
            _BAKER_Q, "96", "12 * 8", _BAKER_POSER["worked"], brain)
        self.assertFalse(ok)
        self.assertIn("unparseable", reason)


class SemanticTickTests(unittest.TestCase):
    def _forge(self, suggest) -> TriviaForge:
        ctx = _ctx(self)
        return TriviaForge(suggest=suggest, db=ctx.db, context=ctx)

    def test_tick_banks_only_semantically_verified(self):
        f = self._forge(_maths_brain(_BAKER_POSER, _BAKER_SOLVER))
        rep = f.refill_bank(max_passages=0, max_maths=0,
                            max_maths_llm=1)
        self.assertEqual(rep["admitted"], 1, rep)
        entry = [e for e in f.bank.items()
                 if e["kind"] == "maths_llm"][0]
        self.assertEqual(entry["answer"], "96")
        self.assertIn("12 * 8", entry["evidence"])
        self.assertIn("multiply", entry["solution"])

    def test_tick_discards_story_mismatch_and_counts_it(self):
        f = self._forge(_maths_brain(
            {"question": _STICKER_Q, "answer": "19",
             "expression": "12 + 4 + 3",
             "worked": "add all the numbers: 12 + 4 + 3"},
            {"answer": "5", "op": "subtract",
             "why": "subtract what she gave away"}))
        rep = f.refill_bank(max_passages=0, max_maths=0,
                            max_maths_llm=1)
        self.assertEqual(rep["admitted"], 0, rep)
        self.assertEqual(rep["discarded"], 1, rep)
        reasons = f.bank_status()["discard_reasons"]
        self.assertTrue(any("independent solve disagrees" in r
                            for r in reasons), reasons)

    def test_brain_absent_procedural_unaffected(self):
        f = self._forge(None)
        rep = f.refill_bank(max_passages=0, max_maths=6,
                            max_maths_llm=2)
        self.assertGreaterEqual(rep["admitted"], 4, rep)
        self.assertTrue(any("maths_llm: no brain" in n
                            for n in rep["notes"]), rep)
        self.assertTrue(all(e["kind"] == "maths"
                            for e in f.bank.items()))


# ── 2. purge behind the owner gate (real registration path) ─────────

from nomorals.core.policy import Capability, CapabilitySet  # noqa: E402
from nomorals.tools.registry import ToolRegistry  # noqa: E402

#: the exact outsider grant the chat brain builds for non-owner
#: surfaces (agents/partner/brain.py): model call + public game
#: playing + read-only files.  No "owner" in it.
OUTSIDER_CAPS = CapabilitySet.of(
    Capability.MODEL_CALL, Capability.SOCIAL_PLAY, Capability.FS_READ)


@pytest.fixture()
def games_reg():
    reg = ToolRegistry()
    reg.register_builtins()
    return reg


def _stocked_forge(reg) -> TriviaForge:
    out = reg.call("games", action="list", actor="owner")
    assert out.ok
    forge = reg._games_engine.forge
    forge.refill_bank(max_passages=0, max_maths=6, max_maths_llm=0)
    assert forge.bank.stock_count() >= 3
    return forge


class TestPurgeGate:
    def test_status_and_refill_stay_public(self, games_reg):
        forge = _stocked_forge(games_reg)
        out = games_reg.call(
            "games", action="trivia_bank", sub="status",
            actor="outsider", capabilities=OUTSIDER_CAPS)
        assert out.ok
        assert out.value["bank"]["stock"] == forge.bank.stock_count()

    def test_purge_refused_by_name_on_public_surface(self, games_reg):
        forge = _stocked_forge(games_reg)
        before = forge.bank.stock_count()
        out = games_reg.call(
            "games", action="trivia_bank", sub="purge",
            actor="outsider", capabilities=OUTSIDER_CAPS)
        assert out.ok  # the tool ran; the ACTION refused
        assert out.value["ok"] is False
        assert "owner-only" in out.value["error"]
        assert "trivia_bank_purge" in out.value["error"]
        assert forge.bank.stock_count() == before

    def test_purge_tool_denied_for_outsider_grant(self, games_reg):
        _stocked_forge(games_reg)
        out = games_reg.call(
            "trivia_bank_purge", actor="outsider",
            capabilities=OUTSIDER_CAPS)
        assert not out.ok  # the registry's own capability gate
        # and the restricted catalog never even shows the tool
        names = {s["name"] for s in
                 games_reg.schemas(capabilities=OUTSIDER_CAPS)}
        assert "trivia_bank_purge" not in names
        assert "games" in names

    def test_purge_succeeds_on_owner_surface(self, games_reg):
        forge = _stocked_forge(games_reg)
        stock = forge.bank.stock_count()
        names = {s["name"] for s in games_reg.schemas()}
        assert "trivia_bank_purge" in names
        # one question by text, then the whole bank
        question = forge.bank.items()[0]["q"]
        out = games_reg.call("trivia_bank_purge", question=question,
                             actor="owner")
        assert out.ok and out.value["ok"] is True
        assert out.value["purged"] == 1
        assert forge.bank.stock_count() == stock - 1
        out = games_reg.call("trivia_bank_purge", actor="owner")
        assert out.ok and out.value["purged"] == stock - 1
        assert forge.bank.stock_count() == 0


# ── 3. hunt payout-doc vetting ───────────────────────────────────────

from nomorals.agents import opportunities as O  # noqa: E402
from nomorals.agents.opportunity_hunt import (  # noqa: E402
    VERIFY_BUDGET, WeeklyOpportunityHunt, platform_of, verify_candidate)

CANDIDATE_URL = "https://tasks.example/apply"
PAY_DOC_URL = "https://tasks.example/payments"
FEES_DOC_URL = "https://help.tasks.example/fees"

#: eligible worldwide, long enough — and silent on payouts.
SILENT_PAGE = (
    "Join our contributor program — open worldwide. Work from "
    "anywhere in the world, on your own schedule, testing websites "
    "and apps for clients everywhere. New contributors sign up in "
    "minutes and start their first paid test today. Browse the "
    "task board to see what is open in your country right now.")
PAY_DOC_PAGE = (
    "How you get paid: contributors are paid via Payoneer every "
    "Friday for the previous week's completed tasks, with no "
    "minimum threshold. Payment lands two business days after "
    "each weekly payout run closes its books.")
PAYPAL_DOC_PAGE = (
    "How you get paid: all contributors are paid via PayPal "
    "within seven days of completing each testing cycle. Make "
    "sure your account email matches your profile email so that "
    "payments arrive without any delay or manual review.")


def _hunt_opp(url: str, title: str = "Paid testing gig"):
    return O.Opportunity(
        kind="paid_task", title=title, source=platform_of(url),
        url=url, payout_text="earn $25 per test", effort="low",
        regions=["global"])


class _Fetch:
    def __init__(self, pages: dict) -> None:
        self.pages = pages
        self.calls: list = []

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        return self.pages.get(url, "")


def _links(url: str):
    return [
        (PAY_DOC_URL, "How you get paid"),
        ("https://other.example/payments", "Payments"),
        ("https://tasks.example/about", "About us"),
        (FEES_DOC_URL, "Fees and payouts"),
    ]


class TestPayoutDocs:
    def test_candidate_silent_doc_names_rail_admitted(self):
        pages = {CANDIDATE_URL: SILENT_PAGE, PAY_DOC_URL: PAY_DOC_PAGE}
        fetch = _Fetch(pages)
        v = verify_candidate(_hunt_opp(CANDIDATE_URL), fetch, _links)
        assert v["ok"] is True
        assert v["payout_rail"] == "Payoneer"
        # the evidence names the page that carried the proof
        assert v["payout_evidence_url"] == PAY_DOC_URL
        assert "payoneer" in v["payout_evidence"].lower()
        assert v["payout_docs_checked"] == [PAY_DOC_URL]
        # off-domain payout links are never followed
        assert "https://other.example/payments" not in fetch.calls

    def test_doc_with_only_paypal_dropped_with_doc_reason(self):
        pages = {CANDIDATE_URL: SILENT_PAGE,
                 PAY_DOC_URL: PAYPAL_DOC_PAGE,
                 FEES_DOC_URL: PAYPAL_DOC_PAGE}
        fetch = _Fetch(pages)
        v = verify_candidate(_hunt_opp(CANDIDATE_URL), fetch, _links)
        assert v["ok"] is False
        assert "payout doc found, no qualifying rail" in v["reason"]
        assert v["payout_docs_checked"] == [PAY_DOC_URL, FEES_DOC_URL]

    def test_no_payout_doc_found_reason(self):
        def links(url: str):
            return [("https://other.example/pay", "Payments"),
                    ("https://tasks.example/about", "About us")]

        fetch = _Fetch({CANDIDATE_URL: SILENT_PAGE})
        v = verify_candidate(_hunt_opp(CANDIDATE_URL), fetch, links)
        assert v["ok"] is False
        assert "no payout doc found" in v["reason"]
        assert fetch.calls == [CANDIDATE_URL]

    def test_run_stores_doc_evidence_url(self, tmp_path):
        opp = _hunt_opp(CANDIDATE_URL)
        (tmp_path / "data").mkdir(parents=True)

        class StubHunter:
            data_dir = tmp_path / "data"

            def scan(self, kinds=None):
                return [opp]

        hunt = WeeklyOpportunityHunt(
            hunter=StubHunter(),
            fetch_text=_Fetch({CANDIDATE_URL: SILENT_PAGE,
                               PAY_DOC_URL: PAY_DOC_PAGE}),
            fetch_links=_links)
        out = hunt.run(now=1000.0)
        assert out["entries"] == 1
        entry = hunt.latest()["entries"][0]
        assert entry["rail"] == "Payoneer"
        assert entry["payout_evidence_url"] == PAY_DOC_URL

    def test_doc_fetches_count_against_verify_budget(self, tmp_path):
        opps = [_hunt_opp(f"https://c{i}.example/apply",
                          title=f"Gig {i}") for i in range(4)]
        pages = {}
        links_by_url = {}
        for i, opp in enumerate(opps):
            host = f"c{i}.example"
            pages[opp.url] = SILENT_PAGE
            docs = [f"https://{host}/p{j}" for j in range(3)]
            links_by_url[opp.url] = [(d, "Payments") for d in docs]
            for d in docs:
                pages[d] = PAYPAL_DOC_PAGE
        # each candidate costs 5 fetches: page + link read + 3 docs
        (tmp_path / "data").mkdir(parents=True)
        hunt = WeeklyOpportunityHunt(
            hunter=type("H", (), {
                "data_dir": tmp_path / "data",
                "scan": lambda self, kinds=None: list(opps)})(),
            fetch_text=_Fetch(pages),
            fetch_links=lambda url: links_by_url.get(url, []))
        out = hunt.run(now=1000.0)
        rec = out["run"]
        assert rec["candidates"] == 4
        assert rec["checks"] == 3  # the 4th never starts: budget spent
        assert rec["fetches"] == 15
        assert rec["fetches"] >= VERIFY_BUDGET
        assert rec["dropped_unverifiable"] == 3


# ── 4. bet desk session books ────────────────────────────────────────

from nomorals.agents.betdesk import (  # noqa: E402
    BettingDesk, SessionBookCollector, render_status)

CSV_BOOK = ("id,busted_at,time\n1,2.5,1700000000\n"
            "2,1.0,1700000060\n3,3.25,1700000120\n")


def _drop(desk_dir: Path, name: str, text: str) -> Path:
    inbox = desk_dir / "inbox"
    inbox.mkdir(exist_ok=True)
    f = inbox / name
    f.write_text(text, encoding="utf-8")
    return f


class TestSessionBooks:
    def test_file_collected_once_and_booked(self, tmp_path):
        desk = BettingDesk(None, desk_dir=str(tmp_path))
        _drop(tmp_path, "bustabit__crash.csv", CSV_BOOK)
        rows = desk.import_inbox()
        assert rows[0]["ok"] and rows[0]["added"] == 3
        assert desk.store.count() == 3
        # moved out of the inbox — studied once
        assert not (tmp_path / "inbox" / "bustabit__crash.csv").exists()
        assert (tmp_path / "inbox" / "done"
                / "bustabit__crash.csv").exists()
        # the session book holds the record, and status shows it
        recs = SessionBookCollector(str(tmp_path),
                                    desk.store).records()
        assert len(recs) == 1
        assert recs[0]["ok"] and recs[0]["added"] == 3
        assert recs[0]["provider"] == "bustabit"
        book = desk.session_book()
        assert book["files_booked"] == 1
        assert book["rounds_landed"] == 3
        st = desk.status()
        assert st["session_book"]["rounds_landed"] == 3
        assert "session books" in render_status(st)

    def test_reimport_of_same_file_is_a_noop(self, tmp_path):
        desk = BettingDesk(None, desk_dir=str(tmp_path))
        _drop(tmp_path, "bustabit__crash.csv", CSV_BOOK)
        desk.import_inbox()
        # the owner drops the very same export again
        _drop(tmp_path, "bustabit__crash.csv", CSV_BOOK)
        rows = desk.import_inbox()
        assert rows[0]["already_booked"] is True
        assert rows[0]["added"] == 0
        assert desk.store.count() == 3  # no duplicate rounds
        recs = SessionBookCollector(str(tmp_path),
                                    desk.store).records()
        assert len(recs) == 1  # and no second book record

    def test_corrupt_file_named_failure_ledger_untouched(self, tmp_path):
        desk = BettingDesk(None, desk_dir=str(tmp_path))
        bad = _drop(tmp_path, "stake__crash.jsonl", "{not json at all\n")
        rows = desk.import_inbox()
        assert rows[0]["ok"] is False
        assert rows[0]["detail"]  # the failure is named
        assert desk.store.count() == 0  # ledger untouched
        assert bad.exists()  # left in the inbox for the owner
        book = desk.session_book()
        assert book["files_failed"] == 1
        assert book["failures"][0]["file"] == "stake__crash.jsonl"
        # a second sweep does not re-fail or re-record it
        rows = desk.import_inbox()
        assert rows[0]["already_booked"] is True
        assert len(SessionBookCollector(str(tmp_path),
                                        desk.store).records()) == 1
        # the owner fixes the export in place: new content, new
        # fingerprint — collected as a fresh import
        bad.write_text('{"round_id": "9", "crash": 2.0, '
                       '"ts": 1700000000}\n', encoding="utf-8")
        rows = desk.import_inbox()
        assert rows[0]["ok"] and rows[0]["added"] == 1
        assert desk.store.count() == 1
        book = desk.session_book()
        assert book["files_failed"] == 0
        assert book["files_booked"] == 1
        assert book["rounds_landed"] == 1

    def test_bad_name_recorded_once(self, tmp_path):
        desk = BettingDesk(None, desk_dir=str(tmp_path))
        _drop(tmp_path, "random.csv", CSV_BOOK)
        rows = desk.import_inbox()
        assert rows[0]["ok"] is False
        assert "provider" in rows[0]["detail"]
        rows = desk.import_inbox()
        assert rows[0]["already_booked"] is True
        assert desk.store.count() == 0
        assert len(SessionBookCollector(str(tmp_path),
                                        desk.store).records()) == 1


if __name__ == "__main__":
    unittest.main()
