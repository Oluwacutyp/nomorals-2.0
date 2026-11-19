"""Craft-Anything port (census §3A.18): ToolSpec with effects + consent
gates generalising the command registry.

Port of the design of storytold/craft-anything (an open-source clean-room
AI build engine in pure Rust, dual MIT/Apache-2.0, upstream commit
078cf0c4): crates/engine/src/tool.rs (Effects bitflags, Toolset, Caller,
ToolSpec), crates/engine/src/lib.rs (Consent, CallOptions, Engine::call),
crates/mcp/src/lib.rs (consent_required, allow-run, over-limit evidence),
crates/ai/src/agent.rs (agent loop over the engine's tools).

Failing-first: these tests were written against the survey before the
organ existed; every one of them failed until the organ landed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.craftanything import effects as E
from nomorals.agents.craftanything.agent import CraftAgent
from nomorals.agents.craftanything.attribution import verify_attribution
from nomorals.agents.craftanything.consent import Consent, ConsentGate
from nomorals.agents.craftanything.engine import (
    Caller,
    CallOptions,
    CancelToken,
    CraftEngine,
)
from nomorals.agents.craftanything.parity import run_parity
from nomorals.agents.craftanything.specs import (
    action_effects,
    declared_effects,
)
from nomorals.agents.craftanything.toolsets import Toolset, toolset_for
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "craftanything"
BRACKET_DXF = Path(__file__).resolve().parent / "fixtures" / "cadcraft" / "sample-bracket.dxf"


def _registry():
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    return registry


class EffectsTest(unittest.TestCase):
    def test_flags_are_distinct_bits(self):
        flags = [
            E.MUTATES_SESSION, E.MUTATES_TARGET, E.WRITES_FILESYSTEM,
            E.LAUNCHES_PROCESS, E.ACCESSES_NETWORK, E.CHANGES_UI_STATE,
            E.MAY_DISCARD_DATA, E.EXECUTES_FOREIGN_CODE, E.MUTATES_DEVICE,
            E.IDEMPOTENT, E.CARRIES_TARGET_CONTENT, E.SEND_MESSAGE,
            E.SPENDS_MONEY,
        ]
        self.assertEqual(len(set(flags)), len(flags))
        for flag in flags:
            self.assertTrue(flag & (flag - 1) == 0, flag)

    def test_named_combos_match_upstream(self):
        self.assertEqual(E.EVIDENCE, E.MUTATES_SESSION | E.IDEMPOTENT)
        self.assertEqual(E.NATIVE_EVIDENCE, E.EVIDENCE | E.LAUNCHES_PROCESS)
        self.assertEqual(E.NETWORK_EVIDENCE, E.EVIDENCE | E.ACCESSES_NETWORK)
        self.assertEqual(E.READ_ONLY, E.IDEMPOTENT)

    def test_effect_names_round_trip(self):
        mask = E.WRITES_FILESYSTEM | E.ACCESSES_NETWORK
        names = E.effect_names(mask)
        self.assertIn("writes_filesystem", names)
        self.assertIn("accesses_network", names)
        self.assertEqual(E.effects_from_names(names), mask)
        self.assertEqual(E.effect_names(0), [])
        self.assertEqual(E.effect_names(E.READ_ONLY), ["idempotent"])

    def test_hints(self):
        self.assertTrue(E.read_only_hint(E.READ_ONLY))
        self.assertFalse(E.read_only_hint(E.READ_ONLY | E.WRITES_FILESYSTEM))
        self.assertTrue(E.destructive_hint(E.MAY_DISCARD_DATA))
        self.assertTrue(E.destructive_hint(E.EXECUTES_FOREIGN_CODE))
        self.assertTrue(E.destructive_hint(E.MUTATES_DEVICE))
        self.assertFalse(E.destructive_hint(E.READ_ONLY))
        self.assertTrue(E.open_world_hint(E.LAUNCHES_PROCESS))
        self.assertTrue(E.open_world_hint(E.ACCESSES_NETWORK))
        self.assertFalse(E.open_world_hint(E.READ_ONLY))

    def test_needs_consent(self):
        self.assertTrue(E.needs_consent(E.EXECUTES_FOREIGN_CODE))
        self.assertTrue(E.needs_consent(E.MUTATES_DEVICE))
        # Devon extension: spending money needs consent too.
        self.assertTrue(E.needs_consent(E.SPENDS_MONEY))
        self.assertFalse(E.needs_consent(E.WRITES_FILESYSTEM))
        self.assertFalse(E.needs_consent(E.READ_ONLY))
        self.assertFalse(E.needs_consent(E.SEND_MESSAGE))

    def test_derive_effects_from_kind_and_capability(self):
        self.assertEqual(
            E.derive_effects("read", "", {}), E.READ_ONLY)
        self.assertEqual(
            E.derive_effects("query", "", {}), E.READ_ONLY)
        self.assertEqual(
            E.derive_effects("write", "fs.write", {}),
            E.WRITES_FILESYSTEM | E.MUTATES_TARGET)
        self.assertEqual(
            E.derive_effects("delete", "fs.delete", {}),
            E.WRITES_FILESYSTEM | E.MUTATES_TARGET | E.MAY_DISCARD_DATA)
        self.assertEqual(
            E.derive_effects("exec", "exec.code", {}),
            E.LAUNCHES_PROCESS | E.EXECUTES_FOREIGN_CODE)
        self.assertEqual(
            E.derive_effects("exec", "exec.shell", {}),
            E.LAUNCHES_PROCESS)
        self.assertEqual(
            E.derive_effects("net", "net.out", {}),
            E.ACCESSES_NETWORK | E.IDEMPOTENT)
        self.assertEqual(
            E.derive_effects("io", "social.dm", {}),
            E.SEND_MESSAGE | E.ACCESSES_NETWORK)
        self.assertEqual(
            E.derive_effects("io", "db.write", {}),
            E.MUTATES_SESSION | E.WRITES_FILESYSTEM)

    def test_infer_requested_effects_from_params(self):
        self.assertEqual(
            E.infer_requested_effects("render", {"output": "/tmp/x.png"}),
            E.WRITES_FILESYSTEM)
        self.assertEqual(
            E.infer_requested_effects("inspect", {"sources": ["/tmp/x.dxf"]}), 0)
        self.assertEqual(
            E.infer_requested_effects("inspect", {}), 0)


class ConsentGateTest(unittest.TestCase):
    def test_read_only_always_allowed(self):
        gate = ConsentGate()
        for caller in (Caller.UI, Caller.CLI, Caller.MCP, Caller.AGENT):
            decision = gate.decide("cadcraft", E.READ_ONLY, caller)
            self.assertEqual(decision.decision, "allow")
            self.assertIn("read-only", decision.reason)

    def test_owner_direct_call_allowed_for_consent_effects(self):
        gate = ConsentGate()
        decision = gate.decide(
            "shell", E.EXECUTES_FOREIGN_CODE, Caller.UI)
        self.assertEqual(decision.decision, "allow")
        self.assertIn("owner", decision.reason)
        decision = gate.decide(
            "shell", E.EXECUTES_FOREIGN_CODE, Caller.CLI)
        self.assertEqual(decision.decision, "allow")

    def test_agent_call_needs_consent_and_names_the_effect(self):
        gate = ConsentGate()
        decision = gate.decide(
            "shell", E.EXECUTES_FOREIGN_CODE, Caller.AGENT)
        self.assertEqual(decision.decision, "consent_required")
        self.assertIn("executes_foreign_code", decision.reason)
        decision = gate.decide(
            "shell", E.EXECUTES_FOREIGN_CODE, Caller.MCP)
        self.assertEqual(decision.decision, "consent_required")

    def test_granted_call_allowed(self):
        gate = ConsentGate()
        decision = gate.decide(
            "shell", E.EXECUTES_FOREIGN_CODE, Caller.AGENT,
            consent=Consent.GRANTED)
        self.assertEqual(decision.decision, "allow")
        self.assertIn("granted", decision.reason)

    def test_standing_allow_and_deny(self):
        gate = ConsentGate()
        gate.allow_tool("shell", effects=E.EXECUTES_FOREIGN_CODE)
        decision = gate.decide(
            "shell", E.EXECUTES_FOREIGN_CODE, Caller.AGENT)
        self.assertEqual(decision.decision, "allow")
        self.assertIn("standing rule", decision.reason)
        gate.deny_tool("shell", effects=E.EXECUTES_FOREIGN_CODE)
        decision = gate.decide(
            "shell", E.EXECUTES_FOREIGN_CODE, Caller.AGENT,
            consent=Consent.GRANTED)
        self.assertEqual(decision.decision, "deny")
        self.assertIn("standing rule", decision.reason)

    def test_allow_run_global(self):
        gate = ConsentGate(allow_run=True)
        decision = gate.decide(
            "shell", E.EXECUTES_FOREIGN_CODE, Caller.AGENT)
        self.assertEqual(decision.decision, "allow")
        self.assertIn("allow-run", decision.reason)

    def test_journal_records_every_decision(self):
        gate = ConsentGate()
        gate.decide("cadcraft", E.READ_ONLY, Caller.UI)
        gate.decide("shell", E.EXECUTES_FOREIGN_CODE, Caller.AGENT)
        self.assertEqual(len(gate.journal), 2)
        self.assertEqual(gate.journal[0]["decision"], "allow")
        self.assertEqual(gate.journal[1]["decision"], "consent_required")
        self.assertEqual(gate.journal[1]["tool"], "shell")
        self.assertIn("executes_foreign_code", gate.journal[1]["effects"])

    def test_journal_round_trip(self):
        gate = ConsentGate()
        gate.decide("shell", E.EXECUTES_FOREIGN_CODE, Caller.AGENT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consent.jsonl"
            gate.save_journal(path)
            gate2 = ConsentGate()
            gate2.load_journal(path)
            self.assertEqual(len(gate2.journal), 1)
            self.assertEqual(gate2.journal[0]["tool"], "shell")

    def test_journal_sample_fixture_loads(self):
        gate = ConsentGate()
        count = gate.load_journal(FIXTURES / "journal-sample.jsonl")
        self.assertEqual(count, 2)
        self.assertEqual(gate.journal[0]["tool"], "cadcraft")
        self.assertEqual(gate.journal[0]["decision"], "allow")
        self.assertEqual(gate.journal[1]["tool"], "shell")
        self.assertEqual(gate.journal[1]["decision"], "consent_required")

    def test_corrupt_and_invalid_fixtures_fail_loudly(self):
        import json as _json
        with self.assertRaises(_json.JSONDecodeError):
            _json.loads((FIXTURES / "corrupt.json").read_text(
                encoding="utf-8"))
        invalid = _json.loads((FIXTURES / "invalid.json").read_text(
            encoding="utf-8"))
        # A spec without kind/capability derives to no effects — the
        # engine then treats it as undeclared, never as a guessed set.
        self.assertEqual(
            E.derive_effects(invalid.get("kind", ""),
                             invalid.get("capability", "")), 0)


class SpecsTableTest(unittest.TestCase):
    def test_media_tools_have_declared_effects(self):
        for name in ("cadcraft", "materialcraft", "texturecraft",
                     "anicraft", "figcraft", "soundcraft",
                     "designcraft", "vectorcraft"):
            effects = declared_effects(name)
            self.assertNotEqual(effects, 0, name)
            self.assertTrue(effects & E.WRITES_FILESYSTEM, name)
            self.assertTrue(effects & E.MUTATES_TARGET, name)

    def test_action_effects_are_narrower_than_tool_effects(self):
        for name in ("cadcraft", "materialcraft", "texturecraft",
                     "figcraft", "soundcraft"):
            tool_effects = declared_effects(name)
            inspect_effects = action_effects(name, "inspect")
            self.assertEqual(inspect_effects, E.READ_ONLY, name)
            self.assertEqual(inspect_effects & ~tool_effects, 0, name)

    def test_unknown_tool_has_no_declaration(self):
        self.assertIsNone(declared_effects("no_such_tool"))
        self.assertIsNone(action_effects("no_such_tool", "inspect"))
        self.assertIsNone(action_effects("cadcraft", "no_such_action"))


class ToolsetsTest(unittest.TestCase):
    def test_toolset_ids_and_titles(self):
        self.assertEqual(Toolset.CORE.id, "core")
        self.assertEqual(Toolset.CORE.title, "Essentials")
        self.assertEqual(Toolset.MEDIA.id, "media")
        self.assertEqual(Toolset.from_id("network"), Toolset.NETWORK)
        self.assertIsNone(Toolset.from_id("no_such_toolset"))
        custom = Toolset.custom("my_group")
        self.assertEqual(custom.id, "my_group")
        self.assertEqual(custom.title, "my_group")

    def test_every_tool_in_exactly_one_toolset(self):
        registry = _registry()
        seen: dict[str, str] = {}
        for name, spec in registry._tools.items():
            toolset = toolset_for(name, spec)
            self.assertIsNotNone(toolset, name)
            self.assertNotIn(name, seen, name)
            seen[name] = toolset.id
        self.assertEqual(len(seen), len(registry._tools))
        self.assertGreaterEqual(len(seen), 730)

    def test_media_tools_are_in_media_toolset(self):
        registry = _registry()
        for name in ("cadcraft", "materialcraft", "texturecraft",
                     "figcraft", "soundcraft", "designcraft",
                     "vectorcraft"):
            spec = registry.get(name)
            self.assertIsNotNone(spec, name)
            self.assertEqual(toolset_for(name, spec), Toolset.MEDIA, name)

    def test_system_and_network_assignment(self):
        registry = _registry()
        for name in ("shell_run", "python_run", "execute_plan", "code"):
            spec = registry.get(name)
            self.assertIsNotNone(spec, name)
            self.assertEqual(toolset_for(name, spec), Toolset.SYSTEM, name)
        for name in ("whatsapp_groups", "web_fetch", "web_search"):
            spec = registry.get(name)
            self.assertIsNotNone(spec, name)
            self.assertEqual(toolset_for(name, spec), Toolset.NETWORK, name)


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = _registry()
        # Tool count before probe tools are registered.
        cls.base_count = len(cls.registry._tools)
        cls._probe_calls: list[dict] = []

        def probe_readonly(output: str = "") -> dict:
            cls._probe_calls.append({"output": output})
            return {"reply": "probe readonly ran", "output": output}

        def probe_foreign(output: str = "") -> dict:
            cls._probe_calls.append({"output": output})
            return {"reply": "probe foreign ran", "output": output}

        def probe_big(output: str = "") -> dict:
            return {"reply": "x" * 5000, "output": output}

        cls.registry.register(
            "probe_readonly", probe_readonly,
            description="Test probe: a read-only tool.",
            capability="", kind="read",
            effects=E.READ_ONLY)
        cls.registry.register(
            "probe_foreign", probe_foreign,
            description="Test probe: runs foreign code.",
            capability="exec.code", kind="exec",
            effects=E.LAUNCHES_PROCESS | E.EXECUTES_FOREIGN_CODE)
        cls.registry.register(
            "probe_big", probe_big,
            description="Test probe: returns an over-limit result.",
            capability="", kind="read",
            effects=E.READ_ONLY)
        cls.engine = CraftEngine(cls.registry, max_result_bytes=1024)

    def test_tool_count_is_734_after_craftfonts(self):
        # 730 after the craft-anything port (lane 111) at its base,
        # 731 after the anicraft port (lane 110) landed between that
        # base and this rebase, 732 after the craft-launcher port
        # (lane 112) added the craftlauncher tool, 733 after the
        # craft-libs port (lane 113) added the craftlibs tool, 734
        # after the craft-fonts port (lane 114) added the craftfonts tool.
        self.assertEqual(self.base_count, 734)

    def test_unknown_tool_is_error_not_panic(self):
        result = self.engine.call("no_such_tool", {})
        self.assertFalse(result["ok"])
        self.assertIn("No tool named no_such_tool", result["error"])
        self.assertIn("list_tools", result.get("remediation", ""))

    def test_read_call_executes_the_real_organ(self):
        self.assertTrue(BRACKET_DXF.is_file(), "cadcraft fixture missing")
        engine = CraftEngine(self.registry)
        result = engine.call(
            "cadcraft",
            {"action": "inspect", "sources": [str(BRACKET_DXF)]},
            CallOptions(caller=Caller.UI))
        self.assertTrue(result["ok"], result)
        value = result["result"]
        self.assertEqual(value["entities"], 42)
        self.assertEqual(value["constraints"], 0)
        self.assertIn("reply", value)
        self.assertEqual(result["consent"]["decision"], "allow")
        self.assertIn("read-only", result["consent"]["reason"])

    def test_large_real_result_is_retained_as_evidence(self):
        # The class engine's 1024-byte budget is smaller than the
        # bracket's inspect result (the full DOF report), so the
        # engine retains the real dict as evidence and names it.
        self.assertTrue(BRACKET_DXF.is_file(), "cadcraft fixture missing")
        result = self.engine.call(
            "cadcraft",
            {"action": "inspect", "sources": [str(BRACKET_DXF)]},
            CallOptions(caller=Caller.UI))
        self.assertTrue(result["ok"], result)
        self.assertTrue(result.get("over_limit"), result)
        evidence = self.engine.evidence(result["evidence_id"])
        self.assertIsNotNone(evidence, result)
        self.assertEqual(evidence["result"]["entities"], 42)

    def test_effect_refusal_names_the_missing_effect(self):
        calls_before = len(self._probe_calls)
        result = self.engine.call(
            "probe_readonly",
            {"output": "/tmp/should-not-exist.txt"},
            CallOptions(caller=Caller.UI))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "effect_refused")
        self.assertIn("writes_filesystem", result["error"])
        self.assertEqual(len(self._probe_calls), calls_before,
                         "refused call must not execute")

    def test_readonly_probe_without_output_is_allowed(self):
        calls_before = len(self._probe_calls)
        result = self.engine.call(
            "probe_readonly", {}, CallOptions(caller=Caller.UI))
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(self._probe_calls), calls_before + 1)

    def test_consent_required_names_the_effect(self):
        calls_before = len(self._probe_calls)
        result = self.engine.call(
            "probe_foreign", {},
            CallOptions(caller=Caller.AGENT))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "consent_required")
        self.assertIn("executes_foreign_code", result["details"]["effects"])
        self.assertIn("executes_foreign_code",
                      result["consent"]["reason"])
        self.assertIn("Approve it", result.get("remediation", ""))
        self.assertEqual(result["details"]["tool"], "probe_foreign")
        self.assertEqual(len(self._probe_calls), calls_before,
                         "consent-gated call must not execute")

    def test_granted_consent_executes(self):
        calls_before = len(self._probe_calls)
        result = self.engine.call(
            "probe_foreign", {},
            CallOptions(caller=Caller.AGENT, consent=Consent.GRANTED))
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(self._probe_calls), calls_before + 1)

    def test_owner_direct_call_bypasses_consent_gate(self):
        calls_before = len(self._probe_calls)
        result = self.engine.call(
            "probe_foreign", {},
            CallOptions(caller=Caller.UI))
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(self._probe_calls), calls_before + 1)

    def test_over_limit_result_is_retained_as_evidence(self):
        result = self.engine.call(
            "probe_big", {}, CallOptions(caller=Caller.UI))
        self.assertTrue(result["ok"], result)
        self.assertTrue(result.get("over_limit"))
        evidence_id = result.get("evidence_id")
        self.assertTrue(evidence_id)
        self.assertEqual(result.get("evidence_uri"), f"evidence://{evidence_id}")
        stored = self.engine.evidence(evidence_id)
        self.assertIsNotNone(stored)
        self.assertEqual(len(stored["result"]["reply"]), 5000)
        self.assertIn("retained as evidence", result["result"])

    def test_journal_records_every_call(self):
        before = len(self.engine.journal)
        self.engine.call("probe_readonly", {}, CallOptions(caller=Caller.UI))
        self.engine.call("no_such_tool", {})
        self.assertEqual(len(self.engine.journal), before + 2)
        entry = self.engine.journal[-2]
        self.assertEqual(entry["tool"], "probe_readonly")
        self.assertEqual(entry["caller"], "ui")
        self.assertTrue(entry["ok"])
        self.assertGreaterEqual(entry["duration_ms"], 0)
        # Param values are never journaled, only the digest + keys.
        self.assertNotIn("output", entry.get("param_keys", []))
        self.assertNotIn("output", entry)

    def test_cancel_token_stops_call_before_execution(self):
        calls_before = len(self._probe_calls)
        token = CancelToken()
        token.cancel()
        result = self.engine.call(
            "probe_readonly", {},
            CallOptions(caller=Caller.UI, cancel=token))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "cancelled")
        self.assertEqual(len(self._probe_calls), calls_before)

    def test_progress_reports_start_and_finish(self):
        seen: list[tuple] = []
        result = self.engine.call(
            "probe_readonly", {},
            CallOptions(caller=Caller.UI,
                        progress=lambda p: seen.append((p.fraction, p.message))))
        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(len(seen), 2)
        self.assertEqual(seen[0][0], 0.0)
        self.assertEqual(seen[-1][0], 1.0)

    def test_list_tools_respects_enabled_toolsets(self):
        all_tools = self.engine.list_tools()
        self.assertGreaterEqual(len(all_tools), 730)
        media_only = self.engine.list_tools(toolsets={"media"})
        self.assertTrue(0 < len(media_only) < len(all_tools))
        names = {t["name"] for t in media_only}
        self.assertIn("cadcraft", names)
        self.assertNotIn("shell", names)
        core_only = self.engine.list_tools(toolsets={"core"})
        self.assertIn("craftanything", {t["name"] for t in core_only})

    def test_inspect_reports_effects_and_hints(self):
        info = self.engine.inspect("cadcraft")
        self.assertEqual(info["name"], "cadcraft")
        self.assertIn("writes_filesystem", info["effects"])
        self.assertIn("mutates_target", info["effects"])
        self.assertFalse(info["hints"]["read_only"])
        self.assertEqual(info["toolset"], "media")
        readonly = self.engine.inspect("probe_readonly")
        self.assertTrue(readonly["hints"]["read_only"])
        self.assertIsNone(self.engine.inspect("no_such_tool"))


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = _registry()
        cls.registry.register(
            "probe_foreign",
            lambda output="": {"reply": "probe foreign ran"},
            description="Test probe: runs foreign code.",
            capability="exec.code", kind="exec",
            effects=E.LAUNCHES_PROCESS | E.EXECUTES_FOREIGN_CODE)
        cls.engine = CraftEngine(cls.registry)
        cls.agent = CraftAgent(cls.engine)

    def test_plan_bracket_task_has_three_steps_in_order(self):
        goal = (
            "import tests/fixtures/cadcraft/sample-bracket.dxf with cadcraft, "
            "solve it, then render it to /tmp/craftanything-proof/bracket.png")
        plan = self.agent.plan(goal)
        self.assertEqual([s.tool for s in plan.steps],
                         ["cadcraft", "cadcraft", "cadcraft"])
        self.assertEqual([s.params.get("action") for s in plan.steps],
                         ["import", "solve", "render"])
        self.assertEqual(plan.steps[0].params["sources"],
                         ["tests/fixtures/cadcraft/sample-bracket.dxf"])
        self.assertEqual(plan.steps[2].params["output"],
                         "/tmp/craftanything-proof/bracket.png")
        self.assertEqual(plan.warnings, [])

    def test_run_plan_produces_a_real_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_png = str(Path(tmp) / "bracket.png")
            goal = (
                f"import {BRACKET_DXF} with cadcraft, solve it, "
                f"then render it to {out_png}")
            plan = self.agent.plan(goal)
            # The planner threads the document through the steps: the
            # import step writes the .cadcraft next to the render output,
            # solve reads it back, render reads the solved document.
            self.assertEqual(
                plan.steps[0].params["output"],
                str(Path(tmp) / "bracket.cadcraft"))
            self.assertEqual(
                plan.steps[1].params["sources"],
                [str(Path(tmp) / "bracket.cadcraft")])
            report = self.agent.run(plan, CallOptions(caller=Caller.UI))
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["ran"], 3)
            self.assertEqual(report["refused"], [])
            self.assertTrue(Path(out_png).is_file())
            self.assertGreater(Path(out_png).stat().st_size, 0)
            self.assertGreater(report["duration_ms"], 0)

    def test_plan_refusal_stops_the_run_and_names_it(self):
        plan = self.agent.plan(
            "run probe_foreign with the agent and no consent")
        # The planner cannot know the probe tool; build the plan by hand
        # from the engine's own catalogue to exercise the refusal path.
        from nomorals.agents.craftanything.agent import PlanStep
        plan.steps = [PlanStep(
            tool="probe_foreign", params={},
            rationale="foreign code without consent",
            effects=["executes_foreign_code"])]
        report = self.agent.run(plan, CallOptions(caller=Caller.AGENT))
        self.assertFalse(report["ok"])
        self.assertEqual(report["ran"], 0)
        self.assertEqual(len(report["refused"]), 1)
        self.assertIn("executes_foreign_code", report["refused"][0]["reason"])

    def test_plan_matches_the_plan_fixture(self):
        import json as _json
        expected = _json.loads(
            (FIXTURES / "plan-bracket.json").read_text(encoding="utf-8"))
        plan = self.agent.plan(expected["goal"])
        got = [{"tool": step.tool, "action": step.params.get("action"),
                "params": {key: value for key, value in step.params.items()
                           if key != "action"}}
               for step in plan.steps]
        self.assertEqual(got, expected["steps"])

    def test_plan_unknown_goal_is_honest_empty(self):
        plan = self.agent.plan("florp the zibblewock twice sideways")
        self.assertEqual(plan.steps, [])
        self.assertTrue(plan.warnings)
        self.assertIn("no matching tool", plan.warnings[0])


class ToolSurfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = _registry()

    def test_craftanything_tool_is_registered(self):
        spec = self.registry.get("craftanything")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.capability, "fs.write")

    def test_surface_list_and_inspect(self):
        outcome = self.registry.call(
            "craftanything", actor="test", action="list_tools",
            toolsets=["media"])
        self.assertTrue(outcome.ok, outcome.error)
        names = {t["name"] for t in outcome.value["tools"]}
        self.assertIn("cadcraft", names)
        outcome = self.registry.call(
            "craftanything", actor="test", action="inspect", tool="cadcraft")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertIn("writes_filesystem", outcome.value["effects"])

    def test_surface_call_dispatches_through_the_engine(self):
        outcome = self.registry.call(
            "craftanything", actor="test", action="call",
            tool="cadcraft", params={
                "action": "inspect",
                "sources": [str(BRACKET_DXF)]},
            caller="ui")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertTrue(outcome.value["ok"])
        self.assertGreater(outcome.value["result"]["entities"], 0)

    def test_surface_parity_and_attribution(self):
        outcome = self.registry.call(
            "craftanything", actor="test", action="parity")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.value["failed"], [])
        outcome = self.registry.call(
            "craftanything", actor="test", action="attribution")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.value["failed"], [])


class ParityAndAttributionTest(unittest.TestCase):
    def test_parity_runs_all_cases(self):
        result = run_parity()
        self.assertEqual(result["failed"], [], result)
        self.assertGreaterEqual(result["passed"], 12)
        self.assertEqual(result["passed"], result["total"])

    def test_parity_missing_fixture_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_parity(fixtures_dir=Path(tmp))
            self.assertGreater(len(result["failed"]), 0)

    def test_attribution_verifies_upstream(self):
        result = verify_attribution()
        self.assertEqual(result["failed"], [], result)
        self.assertGreaterEqual(result["checked"], 6)
        self.assertEqual(result["skipped"], [])

    def test_fixtures_exist(self):
        for name in ("spec-readonly.json", "spec-exec.json",
                     "plan-bracket.json", "journal-sample.jsonl",
                     "over-limit.json", "corrupt.json", "invalid.json"):
            self.assertTrue((FIXTURES / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
