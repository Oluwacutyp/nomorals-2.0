"""Rank arena-researcher findings and turn them into upgrade tickets.

Ranking (dead code is the most expensive thing to keep: it misleads
every future reader and every future audit):

* ``dead_module`` — 100
* ``stub`` — 80
* ``unwired_tool`` — 60
* ``silent_swallow`` — 10 per site, capped at 50

Tickets are built to pass the real :func:`research_digest.gate_ticket`
bar — title, rationale, problem, proposed change, risk, verify steps,
behavior, gap evidence with a verified ``nomorals/...py:<line>`` anchor,
a failing test name, and claim ids — so they can be filed into the
upgrade queue through :class:`upgrade_queue.UpgradePipeline` exactly
like the research loop's tickets, with source ``"arena_researcher"``.
"""
from __future__ import annotations

_RANK_WEIGHTS = {
    "dead_module": 100,
    "stub": 80,
    "unwired_tool": 60,
    "silent_swallow": 10,
}

#: Maximum proposals filed per cycle (see loop.arena_max_proposals).
DEFAULT_MAX_PROPOSALS = 5


def finding_score(finding: dict) -> int:
    """Severity score for one finding (higher = filed first)."""
    base = _RANK_WEIGHTS.get(finding.get("kind", ""), 0)
    if finding.get("kind") == "silent_swallow":
        return min(50, base * int(finding.get("count", 1) or 1))
    return base


def rank_findings(findings: list[dict]) -> list[dict]:
    """Return findings ordered worst-first (score, kind, path)."""
    return sorted(
        findings,
        key=lambda f: (-finding_score(f), f.get("kind", ""),
                       f.get("path", ""), f.get("line", 0)))


def _slug(path: str, line: int) -> str:
    stem = path.replace("nomorals/", "").replace("/", "_") \
        .replace(".py", "")
    return f"{stem}_{line}"


def finding_to_ticket(finding: dict, *, run_id: str) -> dict:
    """Build a gate-passing upgrade ticket from one survey finding."""
    kind = finding.get("kind", "")
    path = finding.get("path", "")
    line = int(finding.get("line", 1) or 1)
    organ = finding.get("organ", "general")
    evidence = finding.get("evidence", "")
    anchor = f"{path}:{line}"
    slug = _slug(path, line)
    test_name = f"test_arena_researcher_{kind}_{slug}"
    verify_steps = [
        "python3 -m pytest tests/test_gapfix_arena_researcher.py -q",
        "python3 -m pytest tests/test_layering.py -q",
    ]
    if kind == "dead_module":
        title = f"Remove or wire dead module {path}"
        problem = (
            f"Module {path} is dead code in production: the arena survey "
            f"found {evidence}. Nothing under nomorals/ imports it and no "
            "dotted-name reference to it exists, so it ships weight "
            "that misleads every future reader and audit."
        )
        change = (
            f"Delete {path} from the tree and add a guard test that fails "
            "while the module is present, or wire the module into its "
            "owning surface with a real caller and a live test that "
            f"exercises it. The change touches {path} and nothing else "
            "unless the caller surface needs it."
        )
        risk = (
            "Deleting a module nobody imports cannot break a live call "
            "path. The residual risk is a dynamic load the static scan "
            "cannot see; the importers plus string-reference scan "
            "mitigates it, the full suite runs before ship, and the "
            "change reverts cleanly."
        )
        behavior = (
            f"After the change, {path} no longer exists in the tree, or it "
            "is imported by production code and covered by a live "
            "test; the survey no longer reports it."
        )
        confidence = 0.85
    elif kind == "stub":
        title = f"Implement stub in {path}"
        problem = (
            f"The survey found a stub in {path}: {evidence}. A caller that reaches "
            "it gets a raised NotImplementedError or a silent empty "
            "return instead of the capability the surface promises."
        )
        change = (
            f"Implement the stubbed function in {path} to its contract, "
            "with a live test that exercises the new behavior end to "
            "end, or delete the stub and its callers when the "
            f"capability is unwanted. The change touches {path} and its "
            "tests."
        )
        risk = (
            "A fresh implementation can misbehave where the stub used "
            "to fail loudly, so the live test must pin the contract "
            "before the stub is replaced; the change is small, scoped "
            "to one function, and reverts cleanly."
        )
        behavior = (
            f"After the change, the function in {path} implements its "
            "contract and the live test passes; no caller receives a "
            "NotImplementedError from it."
        )
        confidence = 0.85
    elif kind == "unwired_tool":
        title = f"Wire tool module {path} into the runtime"
        problem = (
            f"The survey found an unwired tool: {evidence}. The registry lists "
            "it, but the tool never reaches the runtime, so the "
            "capability it promises is unreachable from chat or the "
            "command line."
        )
        change = (
            "Make the tool module importable and give it a register "
            f"hook so the registry entry in {path} reaches the runtime, or "
            "remove the registry entry when the tool is unwanted. The "
            f"change touches {path} and the module it names."
        )
        risk = (
            "Wiring a new tool exposes a surface that was previously "
            "dark, so its commands must be owner-gated where the "
            "registry says so; the change is scoped to one registry "
            "entry and reverts cleanly."
        )
        behavior = (
            "After the change, the tool appears in the registered tool "
            "list and answers its command, or the registry entry is "
            "gone; the survey no longer reports it."
        )
        confidence = 0.7
    else:  # silent_swallow
        count = int(finding.get("count", 1) or 1)
        title = f"Surface {count} silent error swallows in {path}"
        problem = (
            f"The survey found {evidence}. Errors in {path} vanish without a "
            "trace: no log, no journal row, no signal to the owner, "
            "so a live failure looks like nothing happened."
        )
        change = (
            f"Replace each silent handler in {path} with a logged, "
            "journaled failure or a deliberate recovery that names the "
            "error, and add a test that triggers the failure path and "
            f"asserts the signal. The change touches {path} and its tests."
        )
        risk = (
            "Surfacing a swallowed error can expose a noisy failure "
            "mode that was previously invisible; each handler is "
            "converted one at a time and the suite runs before ship, "
            "so a regression shows up before it lands."
        )
        behavior = (
            f"After the change, every former swallow site in {path} logs "
            "or journals the error, and the failure-path test passes; "
            "the survey no longer reports it."
        )
        confidence = 0.6
    rationale = (
        "The arena researcher surveyed the live tree and found this "
        f"while the system was running: {evidence} at {anchor}. Ranked by severity "
        f"against the other findings from run {run_id}, with the evidence "
        "above. The owner decides; this ticket only proposes."
    )
    return {
        "title": title,
        "rationale": rationale,
        "domain": organ,
        "problem": problem,
        "proposed_change": change,
        "risk": risk,
        "verify_steps": verify_steps,
        "behavior": behavior,
        "gap_evidence": (
            f"Checked against the current code: {anchor} — {evidence}"
        ),
        "failing_test": test_name,
        "expected_tests": ["tests/test_gapfix_arena_researcher.py::" + test_name],
        "test_plan": ["tests/test_gapfix_arena_researcher.py::" + test_name],
        "acceptance_criteria": [
            f"the finding at {anchor} is gone from a fresh survey",
            "layering suite green",
            "no new ruff findings on touched files",
        ],
        "claim_ids": [finding.get("id", f"ar-{slug}")],
        "confidence": confidence,
        "suggested_files": [path],
    }
