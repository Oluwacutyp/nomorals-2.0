"""Machine-checked parity for the Craft-Anything organ.

A parity TOML lists real fixtures + operations + expected outcomes;
:func:`run_parity` runs every case and fails on any mismatch. A missing
fixture FAILS its case — never a silent skip (the same habit as the
cadcraft/materialcraft/texturecraft organs).

Expectation keys compare by equality, except for suffixed keys:
``_gt`` / ``_ge`` / ``_lt`` / ``_le`` compare numerically, and
``reason_contains`` checks a substring — so a case can pin a threshold
(like "at least 730 tools") without breaking when a later lane adds a
tool.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

__all__ = ["PARITY_TOML", "run_parity"]

PARITY_TOML = Path(__file__).resolve().parent.parent / "craftanything-parity.toml"

_REGISTRY = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_fixture(name: str, fixtures_dir: Path | None) -> Path | None:
    """Resolve a fixture by name. An explicit fixtures_dir pins the
    fixture set — nothing is read outside it, so a missing fixture in
    that dir is a failure, never a silent fall-through to the repo's
    fixtures. Without one, the repo's fixture dirs are searched."""
    path = Path(name)
    if path.is_absolute():
        return path if path.is_file() else None
    if fixtures_dir is not None:
        candidate = fixtures_dir / name
        return candidate if candidate.is_file() else None
    candidates = [
        _repo_root() / "tests" / "fixtures" / "craftanything" / name,
        _repo_root() / "tests" / "fixtures" / "cadcraft" / name,
        _repo_root() / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        from nomorals.tools.registry import ToolRegistry
        _REGISTRY = ToolRegistry(SimpleNamespace())
        _REGISTRY.register_builtins()
    return _REGISTRY


def _facts_for(case: dict, fixture: Path | None,
               fixtures_dir: Path | None) -> dict:
    """Run a case's ops and measure the facts they produce."""
    from . import effects as E
    from .agent import CraftAgent
    from .caller import Caller
    from .consent import ConsentGate
    from .engine import CraftEngine
    from .toolsets import Toolset, toolset_for

    facts: dict = {}
    ops = case.get("ops", [])
    params = case.get("params", {})
    for op in ops:
        if op == "effects_flags":
            flags = [
                E.MUTATES_SESSION, E.MUTATES_TARGET, E.WRITES_FILESYSTEM,
                E.LAUNCHES_PROCESS, E.ACCESSES_NETWORK, E.CHANGES_UI_STATE,
                E.MAY_DISCARD_DATA, E.EXECUTES_FOREIGN_CODE,
                E.MUTATES_DEVICE, E.IDEMPOTENT, E.CARRIES_TARGET_CONTENT,
                E.SEND_MESSAGE, E.SPENDS_MONEY,
            ]
            facts["flags"] = len(flags)
            facts["distinct"] = len(set(flags)) == len(flags)
        elif op == "effects_combos":
            facts["evidence"] = E.EVIDENCE
            facts["read_only"] = E.READ_ONLY
            facts["native_evidence"] = E.NATIVE_EVIDENCE
            facts["network_evidence"] = E.NETWORK_EVIDENCE
        elif op == "effects_hints":
            facts["read_only_hint"] = E.read_only_hint(E.READ_ONLY)
            facts["destructive_hint"] = E.destructive_hint(E.READ_ONLY)
            facts["open_world_hint"] = E.open_world_hint(E.READ_ONLY)
            facts["needs_consent"] = E.needs_consent(E.READ_ONLY)
        elif op == "derive":
            import json as _json
            spec = _json.loads(fixture.read_text(encoding="utf-8"))
            mask = E.derive_effects(
                spec.get("kind", ""), spec.get("capability", ""),
                spec.get("annotations", {}))
            facts["effects"] = E.effect_names(mask)
        elif op == "consent":
            gate = ConsentGate()
            mask = E.effects_from_names(params.get("effects", []))
            caller = Caller(params.get("caller", "ui"))
            decision = gate.decide(params.get("tool", ""), mask, caller)
            facts["decision"] = decision.decision
            facts["reason_contains"] = decision.reason
        elif op == "engine_call":
            engine = CraftEngine(_registry())
            call_params = dict(params.get("call_params", {}))
            if fixture is not None:
                call_params = {
                    key: ([str(fixture) if item == "{fixture}" else item
                           for item in value]
                          if isinstance(value, list) else
                          (str(fixture) if value == "{fixture}" else value))
                    for key, value in call_params.items()
                }
            result = engine.call(params.get("tool", ""), call_params)
            facts["ok"] = bool(result.get("ok"))
            facts["error_code"] = result.get("error_code", "")
            if isinstance(result.get("result"), dict):
                facts.update({
                    key: value
                    for key, value in result["result"].items()
                    if key in ("entities", "constraints", "layers")})
        elif op == "evidence_retention":
            import json as _json
            payload = _json.loads(fixture.read_text(encoding="utf-8"))
            registry = _registry()
            registry.register(
                "probe_overlimit", lambda payload=payload: dict(payload),
                description="Parity probe: returns an over-limit result.",
                capability="", kind="read", effects=E.READ_ONLY)
            engine = CraftEngine(registry, max_result_bytes=1024)
            result = engine.call("probe_overlimit", {})
            facts["over_limit"] = bool(result.get("over_limit"))
            record = engine.evidence(result.get("evidence_id") or "")
            facts["retrieved"] = record is not None and record.get(
                "result") == payload
            facts["reply_len"] = len(payload.get("reply", ""))
        elif op == "toolsets_cover":
            registry = _registry()
            unassigned = 0
            media_missing = []
            for name, spec in registry._tools.items():
                if toolset_for(name, spec) is None:
                    unassigned += 1
            for name in ("cadcraft", "materialcraft", "texturecraft",
                         "figcraft", "soundcraft", "designcraft",
                         "vectorcraft"):
                spec = registry.get(name)
                if spec is None or toolset_for(name, spec) != Toolset.MEDIA:
                    media_missing.append(name)
            facts["unassigned"] = unassigned
            facts["media_missing"] = media_missing
            facts["tools"] = len(registry._tools)
        elif op == "agent_plan":
            engine = CraftEngine(_registry())
            agent = CraftAgent(engine)
            goal = params.get("goal", "")
            if fixture is not None:
                goal = goal.replace("{fixture}", str(fixture))
            plan = agent.plan(goal)
            facts["steps"] = len(plan.steps)
            facts["tools"] = [step.tool for step in plan.steps]
            facts["actions"] = [step.params.get("action")
                                for step in plan.steps]
        elif op == "registry_generalisation":
            registry = _registry()
            facts["tools"] = len(registry._tools)
            facts["all_gettable"] = all(
                registry.get(name) is not None
                for name in list(registry._tools))
            facts["schemas_ok"] = all(
                isinstance(spec.schema().get("parameters"), dict)
                for spec in list(registry._tools.values())[:50])
        else:  # pragma: no cover - unknown ops are a TOML bug
            raise ValueError(f"unknown parity op {op!r}")
    return facts


def _matches(facts: dict, expect: dict) -> list[str]:
    """Return the expectation keys that do not match (empty = pass)."""
    failed: list[str] = []
    for key, wanted in expect.items():
        if key == "reason_contains":
            if wanted not in str(facts.get("reason_contains", "")):
                failed.append(key)
            continue
        for suffix, check in (("_gt", lambda a, b: a > b),
                              ("_ge", lambda a, b: a >= b),
                              ("_lt", lambda a, b: a < b),
                              ("_le", lambda a, b: a <= b)):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                got = facts.get(base)
                if got is None or not check(got, wanted):
                    failed.append(key)
                break
        else:
            if facts.get(key) != wanted:
                failed.append(key)
    return failed


def run_parity(toml_path: str | Path | None = None, *,
               fixtures_dir: str | Path | None = None) -> dict:
    """Run every parity case. Missing fixture = failed case, never skip."""
    path = Path(toml_path) if toml_path is not None else PARITY_TOML
    fixtures = Path(fixtures_dir) if fixtures_dir is not None else None
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        return {"passed": 0, "failed": ["parity-toml-unreadable"],
                "total": 0, "error": str(exc)}
    cases = data.get("cases", [])
    passed: list[str] = []
    failed: list[str] = []
    for case in cases:
        case_id = case.get("id", "?")
        fixture_name = case.get("fixture")
        fixture = None
        if fixture_name:
            fixture = _resolve_fixture(fixture_name, fixtures)
            if fixture is None:
                failed.append(f"{case_id}: missing fixture {fixture_name}")
                continue
        try:
            facts = _facts_for(case, fixture, fixtures)
        except Exception as exc:  # noqa: BLE001 - a case error is a failure
            failed.append(f"{case_id}: {exc}")
            continue
        mismatched = _matches(facts, case.get("expect", {}))
        if mismatched:
            failed.append(
                f"{case_id}: expected {case.get('expect')} "
                f"mismatched {mismatched} (facts {facts})")
        else:
            passed.append(case_id)
    return {"passed": len(passed), "failed": failed,
            "total": len(cases), "passed_ids": passed}
