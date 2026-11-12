"""RE Pass 4 — missions: deterministic planning over the
capability matrix, the journaled runner, unknown-driven
replanning, resume-from-journal, the dynamic rung, and the
mission surfaces (registry tools, chat verbs, nm re).

The provisioning drive is stubbed at the engine seam in these
tests (the machinery-test pattern of the earlier passes): the
pinned downloads are Pass 2's verified territory, and a unit
test must never fetch an engine. Everything else — planning,
inspection over the real parsers and the real light rung,
evidence folding, reconstruction — runs for real over fixture
targets built in the test.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.reverse import ReEngine, provision
from nomorals.agents.reverse.missions import (
    MissionRunner,
    MissionStore,
    classify_question,
    closer_for,
    plan_steps,
)
from nomorals.storage.db import Database
from tests.test_re_android_graph import _write_tree
from tests.test_re_inspectors import build_elf64, build_pe


class _FakeJournal:
    def __init__(self):
        self.entries = []

    def record(self, decision, context="", signals=None, tags=None,
               **kw):
        self.entries.append({"decision": decision,
                             "context": context,
                             "tags": list(tags or [])})
        return len(self.entries)


class _FakeMemory:
    def __init__(self):
        self.items = []

    def remember(self, text, kind="episode", source=""):
        self.items.append({"text": text, "kind": kind,
                           "source": source})
        return len(self.items)


def _no_provision(self):
    """The provisioning drive, stubbed at the engine seam: every
    engine reports unavailable-with-reason, nothing downloads."""
    return {"engines": {
        name: {"engine": name, "state": "unavailable",
               "available": False, "detail": "stubbed in tests"}
        for name in ("jdk", "ghidra", "unblob")}, "ok": False}


class _TmpHome:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="re-mission-")
        return self.dir

    def __exit__(self, *exc):
        return False


def _engine(home: str):
    journal, memory = _FakeJournal(), _FakeMemory()
    engine = ReEngine(
        db=Database(os.path.join(home, "re.db")), journal=journal,
        memory=memory, tools_root=os.path.join(home, "tools"))
    return engine, journal, memory


def _write(home: str, name: str, data: bytes) -> str:
    path = os.path.join(home, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


class PlannerTest(unittest.TestCase):
    def test_classification_by_question_shape(self):
        self.assertEqual(
            classify_question("what does it do when run?"),
            ["dynamic"])
        self.assertIn("deep", classify_question(
            "how does the login function work?"))
        self.assertIn("graph", classify_question(
            "who calls Beta.poke?"))
        self.assertIn("contents", classify_question(
            "what files are inside this image?"))
        self.assertIn("traffic", classify_question(
            "what hosts does it talk to?"))
        self.assertEqual(classify_question("tell me about this"),
                         [])

    def test_plan_matches_live_ghidra_state(self):
        # the plan is a function of the matrix, not of hope: a
        # provision step exists iff the rung is provisionable
        with _TmpHome() as home:
            root = os.path.join(home, "tools")
            state = provision.ghidra_state(Path(root))
            plan = plan_steps("how does it work inside?",
                              ["/x/bin"], tools_root=root)
            rungs = [s["rung"] for s in plan["steps"]]
            if state["state"] == "provisionable":
                self.assertEqual(rungs[0], "provision")
            elif not state["available"]:
                self.assertTrue(any("Ghidra rung is" in n
                                    for n in plan["notes"]))
            self.assertIn("inspect", rungs)
            self.assertEqual(rungs[-1], "reconstruct")

    def test_undecomposable_question_says_so(self):
        with _TmpHome() as home:
            plan = plan_steps("tell me about this", ["/x/bin"],
                              tools_root=os.path.join(home, "t"))
            self.assertTrue(any("did not decompose" in n
                                for n in plan["notes"]))
            self.assertEqual(
                [s["rung"] for s in plan["steps"]],
                ["inspect", "reconstruct"])

    def test_closer_mapping(self):
        from nomorals.agents.reverse.evidence import EvidenceStore

        store = EvidenceStore(Database(":memory:"))
        gh = store.open_unknown(
            "deep behavior is unestablished — the "
            "self-provisioned Ghidra rung closes this",
            subject_sha256="ab" * 32)
        self.assertEqual(closer_for(gh)[0], "ghidra")
        ub = store.open_unknown(
            "contents unestablished — the self-provisioned "
            "unblob rung closes this", subject_sha256="ab" * 32)
        self.assertEqual(closer_for(ub)[0], "unblob")
        dyn = store.open_unknown(
            "what the payload does in motion",
            subject_sha256="ab" * 32,
            required_authority="controlled-replay",
            required_confidence="observed")
        self.assertEqual(closer_for(dyn)[0], "dynamic")
        plain = store.open_unknown(
            "no debug sections are present",
            subject_sha256="ab" * 32,
            required_authority="historical-reference")
        self.assertEqual(closer_for(plain)[0], "none")


@mock.patch.object(ReEngine, "provision_engines", _no_provision)
class MissionRunTest(unittest.TestCase):
    def test_mission_end_to_end_over_elf(self):
        with _TmpHome() as home:
            engine, journal, memory = _engine(home)
            target = _write(home, "sample", build_elf64())
            res = engine.run_mission(
                "what is this file and what does it link against?",
                [target])
            self.assertEqual(res["status"], "closed", res)
            self.assertTrue(all(
                s["status"] == "done" for s in res["steps"]))
            report = res["report"]["targets"][target]
            texts = [c["text"] for c in report["claims"]]
            self.assertTrue(any("libfoo.so.1" in t for t in texts),
                            texts)
            self.assertTrue(any(c["evidence_ids"]
                                for c in report["claims"]))
            # the close journaled + remembered through the seams
            # (the seam formats "RE inspection: <name>" — the
            # mission's name rides inside it)
            self.assertTrue(any(
                "RE mission:" in e["decision"]
                for e in journal.entries))
            self.assertTrue(any(
                "RE mission:" in i["text"]
                for i in memory.items))
            # and the journal reads back: list + status
            listed = engine.mission_list()
            self.assertEqual(len(listed["missions"]), 1)
            got = engine.mission(res["mission_id"])
            self.assertEqual(got["status"], "closed")
            self.assertIn(target, got["report"]["targets"])

    def test_dynamic_question_opens_unknown_naming_missing_engines(self):
        # the gate is dissolved: with no dynamic rung available
        # on this device, the question opens a dynamic_required
        # unknown naming the CONCRETE missing engines — never a
        # gate — and the mission still works every static rung
        with _TmpHome() as home:
            engine, _, _ = _engine(home)
            target = _write(home, "sample", build_elf64())
            unavailable = {
                "frida": {"available": False,
                          "state": "unavailable",
                          "detail": "frida is not provisioned on "
                                    "this machine yet"},
                "gdb": {"available": False,
                        "state": "unavailable",
                        "detail": "gdb is not installed on this "
                                  "machine yet"},
                "sandbox": {"available": False,
                            "state": "unavailable",
                            "detail": "no namespace sandbox works "
                                      "on this device"},
                "available": False, "rung": ""}
            with mock.patch(
                    "nomorals.agents.reverse.dynamic.dynamic_state",
                    return_value=unavailable):
                res = engine.run_mission(
                    "what does it do when run, and what side "
                    "effects does it have?", [target])
            self.assertEqual(res["status"], "closed", res)
            # no rung that executes anything was ever planned
            self.assertTrue(all(
                s["rung"] in ("inspect", "reconstruct")
                for s in res["steps"]))
            sha = res["steps"][0]["detail"]["sha256"]
            unknowns = engine.store.list_unknowns(
                status="open", limit=100)
            dynamic = [u for u in unknowns
                       if u.subject_sha256 == sha
                       and u.required_authority
                       == "controlled-replay"
                       and u.required_confidence == "observed"]
            self.assertEqual(len(dynamic), 1)
            self.assertIn("when it runs", dynamic[0].question)
            self.assertIn("frida", dynamic[0].question)
            self.assertIn("gdb", dynamic[0].question)
            self.assertNotIn("sandbox/consent",
                             dynamic[0].question)
            # the mission report parks it with the concrete
            # missing engines named, not a gate
            parked = res["report"].get("parked_unknowns") or []
            self.assertTrue(any(
                p["unknown_id"] == dynamic[0].unknown_id
                and "frida" in p["reason"]
                for p in parked), parked)
            # static claims still landed — the missing rung
            # blocks the dynamic answer, not the mission
            report = res["report"]["targets"][target]
            self.assertTrue(report["claims"])

    def test_dynamic_question_plans_a_dynamic_step_when_available(self):
        # the plan is a function of the matrix (the
        # test_plan_matches_live_ghidra_state pattern): with a
        # dynamic rung available, the question plans a dynamic
        # step — the execution itself is proven in
        # test_re_dynamic.py, so the engine seam is mocked here
        with _TmpHome() as home:
            engine, _, _ = _engine(home)
            target = _write(home, "sample", build_elf64())
            fake = {"rung": "gdb", "exit_code": 0,
                    "timed_out": False, "duration_s": 0.1,
                    "sandbox": {"backend": "unshare",
                                "network_enforced": True},
                    "subject": {"sha256": "ab" * 32},
                    "evidence_ids": ["ev-dynamic"],
                    "unknowns": [], "card_text": "dynamic card"}
            with mock.patch.object(ReEngine, "run_dynamic",
                                   return_value=fake):
                res = engine.run_mission(
                    "what does it do when run?", [target])
            self.assertEqual(res["status"], "closed", res)
            dynamic_steps = [s for s in res["steps"]
                             if s["rung"] == "dynamic"]
            self.assertTrue(dynamic_steps, res["steps"])
            self.assertTrue(all(
                s["status"] == "done" for s in dynamic_steps))
            self.assertIn("ev-dynamic",
                          dynamic_steps[0]["evidence_ids"])

    def test_interruption_and_resume_from_journal(self):
        with _TmpHome() as home:
            engine, _, _ = _engine(home)
            target = _write(home, "sample", build_elf64())
            runner = MissionRunner(engine)
            first = runner.start(
                "what is this file?", [target], max_steps=1)
            self.assertEqual(first["status"], "running")
            steps = first["steps"]
            self.assertEqual(len(steps), 2)
            self.assertEqual(steps[0]["status"], "done")
            self.assertEqual(steps[1]["status"], "pending")
            # a fresh runner over the same journal resumes it —
            # no state lived in the first runner
            resumed = engine.resume_mission(first["mission_id"])
            self.assertEqual(resumed["status"], "closed")
            self.assertEqual(len(resumed["steps"]), 2)
            self.assertEqual(
                [s["seq"] for s in resumed["steps"]], [0, 1])
            self.assertTrue(all(
                s["status"] == "done" for s in resumed["steps"]))

    def test_unknown_drives_provision_step(self):
        # a PE target cannot be answered by the light rung, so
        # the first inspection leaves the Ghidra-gap unknown; the
        # replan must schedule the rung that closes it (provision
        # first — stubbed here) instead of dropping the question
        with _TmpHome() as home:
            engine, _, _ = _engine(home)
            target = _write(home, "sample.exe", build_pe())
            res = engine.run_mission("what is this file?", [target])
            self.assertEqual(res["status"], "closed", res)
            provision_steps = [
                s for s in res["steps"] if s["rung"] == "provision"]
            self.assertTrue(provision_steps, res["steps"])
            self.assertTrue(any(
                "unknown-driven" in str(s["detail"].get("reason", ""))
                or "unknown" in str(s["detail"].get("reason", ""))
                for s in res["steps"]
                if s["rung"] == "inspect"), res["steps"])
            # the stubbed drive did not land, so the gap unknown
            # honestly remains open — the mission reports, not hides
            sha = next(s["detail"]["sha256"] for s in res["steps"]
                       if s["rung"] == "inspect")
            gaps = [u for u in engine.store.list_unknowns(
                status="open", limit=100)
                if u.subject_sha256 == sha
                and "Ghidra" in u.question]
            self.assertTrue(gaps)

    def test_graph_mission_over_jadx_tree(self):
        with _TmpHome() as home:
            engine, _, _ = _engine(home)
            root = Path(home) / "jadx"
            root.mkdir()
            _write_tree(root)
            res = engine.run_mission(
                "who calls Beta.poke in this app?", [str(root)],
                graph_target="Beta.poke")
            self.assertEqual(res["status"], "closed", res)
            rungs = [s["rung"] for s in res["steps"]]
            self.assertIn("graph", rungs)
            answers = [r for r in
                       engine.store.list_recent(limit=100)
                       if r.predicate_type == "android.references"]
            self.assertTrue(answers)
            self.assertTrue(any(
                "Alpha" in str(a.result) for a in answers))


@mock.patch.object(ReEngine, "provision_engines", _no_provision)
class MissionSurfacesTest(unittest.TestCase):
    def test_registry_mission_actions(self):
        from nomorals.agents import reverse
        from nomorals.tools.registry import ToolRegistry

        context = SimpleNamespace(db=Database(":memory:"),
                                  memory=None, settings=None)
        registry = ToolRegistry(context, enforce=False)
        reverse.register(registry)
        with _TmpHome() as home:
            target = _write(home, "sample", build_elf64())
            outcome = registry.call(
                "re_inspect", action="mission",
                question="what is this file?", targets=target)
            self.assertTrue(outcome.ok, outcome)
            self.assertEqual(outcome.value["status"], "closed")
            mission_id = outcome.value["mission_id"]
            listed = registry.call("re", action="missions")
            self.assertTrue(listed.ok, listed)
            self.assertEqual(len(listed.value["missions"]), 1)
            got = registry.call("re", action="mission",
                                mission_id=mission_id)
            self.assertTrue(got.ok, got)
            self.assertEqual(got.value["mission_id"], mission_id)
            resumed = registry.call(
                "re_inspect", action="mission", mission_id=mission_id)
            self.assertTrue(resumed.ok, resumed)
            self.assertEqual(resumed.value["status"], "closed")

    def test_cli_and_chat_mission_verbs(self):
        from nomorals.agents.partner.runtime_games import (
            RuntimeGamesMixin,
        )
        from nomorals.cmdline.commands.re import _cmd_re

        context = SimpleNamespace(db=Database(":memory:"),
                                  memory=None, settings=None)
        with _TmpHome() as home:
            target = _write(home, "sample", build_elf64())
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_re(SimpleNamespace(
                    re_action="mission", question="what is this?",
                    targets=["--", target]), context)
            self.assertEqual(rc, 0)
            self.assertIn("RE mission", buf.getvalue())
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_re(SimpleNamespace(
                    re_action="missions"), context)
            self.assertEqual(rc, 0)
            self.assertIn("RE missions", buf.getvalue())

        class _RT(RuntimeGamesMixin):
            pass

        rt = _RT()
        rt.context = SimpleNamespace(db=Database(":memory:"),
                                     memory=None, settings=None)
        with _TmpHome() as home:
            target = _write(home, "sample", build_elf64())
            card = rt._control_re(
                f"mission what is this file? -- {target}")
            self.assertIn("RE mission", card)
            listing = rt._control_re("missions")
            self.assertIn("RE missions", listing)

    def test_parser_parses_mission_commands(self):
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        args = parser.parse_args(
            ["re", "mission", "what is this?", "--", "/x", "/y"])
        self.assertEqual(args.re_action, "mission")
        self.assertEqual(args.targets, ["/x", "/y"])
        args = parser.parse_args(["re", "missions"])
        self.assertEqual(args.re_action, "missions")
        args = parser.parse_args(["re", "mission-status", "ms:a"])
        self.assertEqual(args.re_action, "mission-status")
        args = parser.parse_args(["re", "mission-resume", "ms:a"])
        self.assertEqual(args.re_action, "mission-resume")

    def test_mission_store_prefix_lookup(self):
        store = MissionStore(Database(":memory:"))
        made = store.create(
            question="q?", targets=["/x"],
            plan={"intents": [], "dynamic": False, "notes": [],
                  "steps": [{"rung": "inspect", "action": "inspect",
                             "target": "/x", "detail": {}}]})
        self.assertEqual(store.get(made["mission_id"])["question"],
                         "q?")
        short = made["mission_id"][:8]
        self.assertEqual(store.get(short)["mission_id"],
                         made["mission_id"])
        self.assertEqual(store.counts().get("planned"), 1)


if __name__ == "__main__":
    unittest.main()
