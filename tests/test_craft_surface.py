"""/craft — the docs 22+23 craft chat surface, pinned.

The Stage A–C machinery shipped as library + renders; this suite
pins the owner surface that wires it into the command catalog:

* the board renders from seeded records — a domain whose records
  earn a level shows that level WITH a basis line, a greenfield
  domain is named greenfield (not hidden, not zeroed), pack
  states and open items ride along;
* /craft is owner-only at the gating layer (CONTROL_COMMANDS,
  never PUBLIC_COMMANDS — the public count does not move) and at
  the runtime choke point;
* /craft intake renders the gate's verdicts in the seam's own
  words and executes NOTHING — the delegation ledger and claim
  counts are byte-identical before/after;
* /craft status shows a live temporary grant with its expiry and
  benchmark citation, and an expired grant as unmade-with-reason;
* no internal ids in any rendering (contract ids, grant ids,
  uuid shapes) — pinned by direct checks and regex over outputs.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import plan_contracts
from nomorals.agents.delegation import DelegationStore
from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.craft import CoverageReport, render_board
from nomorals.craft.composer import compose_plan
from nomorals.craft.pilots import run_cbt_pilot, run_software_pilot
from nomorals.craft.temporary import (
    CompetenceGrantStore,
    acquire_competence,
)
from nomorals.skills.filebook import FileSkillLibrary
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    caller_may_run,
    detailed_help,
    parse_control,
)
from nomorals.storage.db import Database


def _tmp(testcase: unittest.TestCase) -> Path:
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR")))
    testcase.addCleanup(
        lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
    return tmp


def _tmp_db(testcase: unittest.TestCase) -> Database:
    db = Database(str(_tmp(testcase) / "test.db"))
    db.migrate()
    return db


def _seed_library(user_root: Path) -> FileSkillLibrary:
    # user_dir MUST be per-test: a fixed shared root leaks state
    # across tests and across runs (a temp pack minted by one test
    # becomes another test's "recorded pack", and the directory
    # outlives the run) — the seed side stays the shipped library.
    return FileSkillLibrary(user_dir=user_root)


def _empty_library(tmp: Path) -> FileSkillLibrary:
    (tmp / "seed").mkdir(parents=True, exist_ok=True)
    return FileSkillLibrary(seed_dir=tmp / "seed",
                            user_dir=tmp / "user")


class CraftRuntime(RuntimeSystemMixin):
    """The system mixin alone, with craft stores injected the way
    the handler reads them."""

    def __init__(self, db, library, grants):
        self.context = SimpleNamespace(db=db)
        self._craft_library = library
        self._craft_grants = grants


_SOFTWARE_TASK = {
    "title": "fix the add function",
    "goal": "make add() add, proven by its acceptance test",
    "files": {"calc.py": "def add(a, b):\n    return a - b\n"},
    "change": {"calc.py": "def add(a, b):\n    return a + b\n"},
    "tests": {"test_calc.py": "from calc import add\n\n\n"
                              "def test_add():\n"
                              "    assert add(2, 3) == 5\n"},
}

_CBT_EXERCISE = {
    "kind": "behavioral_activation",
    "fields": {"activity": "A 20-minute walk after lunch",
               "scheduled_for": "Tomorrow at 13:00",
               "expected_pleasure": 6, "expected_mastery": 5},
}

_ID_SHAPES = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}|cg-[0-9a-f]{6,}")


def _pilots(testcase: unittest.TestCase):
    tmp = _tmp(testcase)
    db = _tmp_db(testcase)
    library = _seed_library(tmp / "user")
    grants = CompetenceGrantStore(tmp / "grants.json")
    run_software_pilot(
        db, workspace=tmp / "ws", blob_dir=tmp / "blobs",
        task=dict(_SOFTWARE_TASK), library=library, fixtures=())
    run_cbt_pilot(
        db, exercise=dict(_CBT_EXERCISE),
        interventions_path=tmp / "interventions.db",
        library=library)
    return tmp, db, library, grants


# ── the board ────────────────────────────────────────────────────


class BoardTests(unittest.TestCase):

    def test_board_renders_derived_levels_with_basis(self):
        _tmp_, db, library, grants = _pilots(self)
        rt = CraftRuntime(db, library, grants)
        board = rt._control_craft("")
        self.assertIn("craft — the cooperative board", board)
        # health's records earn L2, and the earning facts ride
        self.assertIn("Health — level L2 (tool-executed)", board)
        self.assertIn("tool-executed: a contract for the field "
                      "completed with supported evidence", board)
        # build's live pack earns L0, pack state named
        self.assertIn("Build — level L0 (knowledge)", board)
        self.assertIn("packs: software-craft-pack (live, valid)",
                      board)
        # open items are counted, not implied
        self.assertIn("open items:", board)
        self.assertIn("open items across the board:", board)
        # the full board is also the bare command + the board verb
        self.assertEqual(board, rt._control_craft("board"))

    def test_board_names_greenfield_and_absences(self):
        _tmp_, db, library, grants = _pilots(self)
        board = CraftRuntime(db, library, grants)._control_craft("")
        # hospitality is seed-known only: named greenfield, in
        # words, on the board — never hidden, never a quiet zero
        self.assertIn("Hospitality", board)
        self.assertIn("greenfield", board)
        self.assertIn("absent — pack: no pack recorded", board)

    def test_coverage_verb_renders_the_measured_detail(self):
        _tmp_, db, library, grants = _pilots(self)
        text = CraftRuntime(
            db, library, grants)._control_craft("coverage")
        self.assertIn("craft coverage — measured state only", text)
        self.assertIn("Health", text)
        self.assertIn("level L2 (tool-executed)", text)
        self.assertIn("contracts: 1 on record", text)

    def test_board_on_fresh_stores_is_honest(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        library = _empty_library(tmp)
        grants = CompetenceGrantStore(tmp / "grants.json")
        board = CraftRuntime(db, library, grants)._control_craft("")
        self.assertIn("craft — the cooperative board", board)
        # the taxonomy still names its domains; each reads
        # unlevelled with its absences in words
        self.assertIn("no level claimed", board)
        self.assertIn("greenfield", board)
        self.assertIn("absent — pack: no pack recorded", board)

    def test_empty_report_renders_the_named_empty_board(self):
        text = render_board(CoverageReport())
        self.assertIn("the board is empty", text)
        self.assertIn("no domain is known to any record", text)

    def test_unknown_verb_and_missing_args_answer_with_usage(self):
        _tmp_, db, library, grants = _pilots(self)
        rt = CraftRuntime(db, library, grants)
        self.assertIn("unknown /craft verb 'frobnicate'",
                      rt._control_craft("frobnicate"))
        self.assertIn("usage: /craft status <field>",
                      rt._control_craft("status"))
        self.assertIn("usage: /craft intake <goal>",
                      rt._control_craft("intake"))


# ── intake: the seam's words, and nothing executes ───────────────


class IntakeTests(unittest.TestCase):

    def test_intake_verdicts_are_the_seams_own_words(self):
        _tmp_, db, library, grants = _pilots(self)
        rt = CraftRuntime(db, library, grants)
        text = rt._control_craft(
            "intake studio: design a poster series")
        self.assertIn("intake — design a poster series", text)
        self.assertIn("pattern: studio", text)
        self.assertIn("decomposition:", text)
        self.assertIn("team (role contracts):", text)
        self.assertIn("gate (the shipped consequential rule):",
                      text)
        self.assertIn("risks:", text)
        self.assertIn("cost (role-contract budgets):", text)
        # the same plan, judged by the seam directly: every reason
        # the seam names appears in the render VERBATIM
        plan = compose_plan("design a poster series",
                            preset="studio")
        verdicts = dict()
        for name, fires, reasons in plan.consequentiality():
            verdicts[name] = (fires, reasons)
        self.assertTrue(verdicts["review"][0])
        for reason in verdicts["review"][1]:
            self.assertIn(reason, text)
        self.assertIn("a delegation contract opens before it runs",
                      text)
        self.assertIn("no contract — the step meets none", text)
        # the read-only boundary is stated on the reply itself
        self.assertIn("preview only", text)
        self.assertIn("opened no contract", text)

    def test_intake_resolves_domain_and_governing_pack(self):
        _tmp_, db, library, grants = _pilots(self)
        rt = CraftRuntime(db, library, grants)
        text = rt._control_craft("intake ship the build pipeline")
        self.assertIn("governed by pack software-craft-pack", text)
        self.assertIn("domain standing: Build at level L0", text)

    def test_intake_executes_nothing(self):
        _tmp_, db, library, grants = _pilots(self)
        rt = CraftRuntime(db, library, grants)
        store = DelegationStore(db)

        def ledger_state():
            contracts = store.recent_contracts(limit=200)
            claims = sum(len(store.claims(c.id, limit=1000))
                         for c in contracts)
            return [(c.id, c.status) for c in contracts], claims

        before = ledger_state()
        rt._control_craft("intake studio: design a poster series")
        rt._control_craft("intake ship the build pipeline")
        rt._control_craft("intake research a brand new field")
        after = ledger_state()
        # no contract opened, none settled, no claim written —
        # the ledger is exactly what it was
        self.assertEqual(before, after)

    def test_intake_verdicts_match_the_rule_not_a_rewording(self):
        # plan_contracts is the authority: a step the rule fires on
        # is rendered as opening a contract, with the rule's
        # reasons joined in ITS words — pinned against the seam
        # function itself, so a handler-side paraphrase fails here.
        plan = compose_plan("design a poster series",
                            preset="studio")
        for name, fires, reasons in plan.consequentiality():
            live = plan_contracts.step_consequentiality(
                plan.step(name))
            self.assertEqual((fires, list(reasons)),
                             (live[0], live[1]))


# ── status: claims from records ──────────────────────────────────

_PASSING_FIXTURE = "def test_field_basics():\n    assert 1 + 1 == 2\n"

_SECTIONS = {
    "terminology": "Hive, apiary, frame, smoker.",
    "workflow": "Inspect, record, treat, review.",
    "tools": "Smoker, hive tool, frames.",
    "rubric": "A claim is proven only by its benchmark.",
    "anti_patterns": "Never claim a field from prose alone.",
    "exemplars": "The benchmark fixture in this pack.",
    "evaluators": "The fixtures declared on this pack.",
    "failure_modes": "Benchmark theatre: fixtures that cannot "
                     "fail prove nothing.",
}


def _research(field: str) -> dict:
    return {"summary": f"research findings for {field}",
            "sections": dict(_SECTIONS),
            "fixtures": [{"filename": "test_benchmark.py",
                          "content": _PASSING_FIXTURE}]}


class StatusTests(unittest.TestCase):

    def _setup(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        library = _empty_library(tmp)
        grants = CompetenceGrantStore(tmp / "grants.json")
        return tmp, db, library, grants

    def test_status_live_pack_claim(self):
        _tmp_, db, library, grants = _pilots(self)
        text = CraftRuntime(
            db, library, grants)._control_craft("status build")
        self.assertIn("craft status — build", text)
        self.assertIn("claimed from a live pack", text)
        self.assertIn("level L0 (knowledge)", text)
        self.assertIn("software-craft-pack", text)

    def test_status_live_grant_shows_expiry_and_benchmark(self):
        tmp, db, library, grants = self._setup()
        rt = CraftRuntime(db, library, grants)
        run = acquire_competence(
            db, library, field="beekeeping",
            research_fn=_research, blob_dir=tmp / "blobs",
            grants=grants, now=time.time())
        self.assertTrue(run.claim_made, run.detail)
        text = rt._control_craft("status beekeeping")
        self.assertIn("claimed temporarily", text)
        self.assertIn("level L0 (knowledge)", text)
        self.assertIn("grade G1", text)
        # the expiry, formatted from the recorded grant
        when = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(run.grant.expires_at))
        self.assertIn(f"lapses at {when}", text)
        # the benchmark citation the claim stands on
        self.assertIn(run.grant.benchmark_ref, text)
        self.assertIn("benchmark passed and is on record", text)
        # the grant's own id never prints
        self.assertNotIn(run.grant.id, text)

    def test_status_expired_grant_is_unmade_with_reason(self):
        tmp, db, library, grants = self._setup()
        rt = CraftRuntime(db, library, grants)
        run = acquire_competence(
            db, library, field="beekeeping",
            research_fn=_research, blob_dir=tmp / "blobs",
            grants=grants, ttl_seconds=600, now=1000.0)
        self.assertTrue(run.claim_made, run.detail)
        text = rt._control_craft("status beekeeping")
        self.assertIn("no claim stands for this field", text)
        self.assertIn("lapsed", text)
        self.assertNotIn("claimed temporarily", text)

    def test_status_unknown_field_is_unmade_with_reason(self):
        _tmp_, db, library, grants = _pilots(self)
        text = CraftRuntime(
            db, library, grants)._control_craft(
                "status beekeeping")
        self.assertIn("no claim stands for this field", text)
        self.assertIn("no pack is recorded for this field", text)


# ── no internal ids, anywhere ────────────────────────────────────


class NoInternalIdsTests(unittest.TestCase):

    def test_no_rendering_carries_internal_ids(self):
        tmp, db, library, grants = _pilots(self)
        rt = CraftRuntime(db, library, grants)
        run = acquire_competence(
            db, library, field="beekeeping",
            research_fn=_research, blob_dir=tmp / "blobs2",
            grants=grants, now=time.time())
        outputs = [
            rt._control_craft(""),
            rt._control_craft("coverage"),
            rt._control_craft(
                "intake studio: design a poster series"),
            rt._control_craft("intake ship the build pipeline"),
            rt._control_craft("status build"),
            rt._control_craft("status beekeeping"),
        ]
        store = DelegationStore(db)
        contract_ids = [c.id for c in
                        store.recent_contracts(limit=200)]
        self.assertTrue(contract_ids)
        for text in outputs:
            self.assertIsNone(_ID_SHAPES.search(text), text)
            for cid in contract_ids:
                self.assertNotIn(cid, text)
            self.assertNotIn(run.grant.id, text)


# ── gating: owner-only, end to end ───────────────────────────────


class GatingTests(unittest.TestCase):

    def test_gating_layer(self):
        self.assertIn("craft", CONTROL_COMMANDS)
        # owner-only: never on the public list, and the public
        # count is exactly what it was before this surface
        self.assertNotIn("craft", PUBLIC_COMMANDS)
        self.assertEqual(len(PUBLIC_COMMANDS), 248)
        self.assertFalse(caller_may_run("craft", False))
        self.assertTrue(caller_may_run("craft", True))
        cmd = parse_control("/craft coverage")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "craft")

    def test_public_help_serves_no_craft_page(self):
        page = detailed_help("craft", public_only=True)
        self.assertNotIn("the craft cooperative board", page)
        owner_page = detailed_help("craft")
        self.assertIn("the craft cooperative board", owner_page)


class RuntimeDispatchTests(unittest.TestCase):
    """The full PartnerRuntime path: /craft dispatches for the
    owner and is refused for anyone else at the grant choke
    point."""

    @classmethod
    def setUpClass(cls):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self, name):
                super().__init__(media_dir="/tmp/nm-test-media")
                self.name = name

            def run(self, handler):
                pass

            def send(self, chat, text, *, reply_to="", buttons=None,
                     **kwargs):
                return SendResult(ok=True, platform=self.name,
                                  message_id="m1")

        tmp = tempfile.TemporaryDirectory(
            prefix="nm-test-craft-surface-")
        cls._tmp = tmp
        settings = load_settings(overrides={
            "home": tmp.name,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)
        gateway = ChatGateway({"telegram": FakeAdapter("telegram")},
                              db=context.db)
        cls.runtime = PartnerRuntime(context, gateway=gateway)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_owner_dispatch_renders_the_board(self):
        reply = self.runtime.handle_control(
            "/craft", "telegram:1", message=None)
        self.assertIn("craft — the cooperative board", reply)
        self.assertNotIn("unknown command", reply)

    def test_non_owner_is_refused_at_the_grant(self):
        with mock.patch.object(self.runtime, "_is_operator",
                               return_value=False):
            reply = self.runtime.handle_control(
                "/craft", "telegram:999",
                message=SimpleNamespace())
        self.assertNotIn("cooperative board", reply)
        self.assertIn("owner", reply)


if __name__ == "__main__":
    unittest.main()
