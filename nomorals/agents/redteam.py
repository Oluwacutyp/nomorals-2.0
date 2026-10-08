"""AI red-teaming / self-attack module — DEFENSIVE security testing.

This module plays the attacker against Devon's OWN agent loop, in a
sandbox, to find holes in OUR defenses before anyone else does.  Every
attack runs against scripted fakes and mocked tools: no external
systems are touched, no real credentials exist here, no destructive
action is ever executed for real.

This is NOT :mod:`nomorals.tools.attacker` — that tool performs
authorized offensive-security work (pentest-style) against EXTERNAL
targets at the owner's direction.  This module does the opposite: it
attacks Devon herself, so we can harden her.

Scenario semantics: a scenario "succeeds" when the ATTACK worked, which
is BAD for us — ``AttackFinding.succeeded=True`` means we have a hole.
``RedTeamReport.failed`` counts those holes.  Anything the harness
cannot decide is recorded ``inconclusive`` (fail-closed: never a false
"passed").
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.result import Err, Ok

_log = logging.getLogger(__name__)

__all__ = [
    "AttackScenario",
    "AttackFinding",
    "RedTeamReport",
    "RedTeam",
    "SCENARIOS",
    "build_scenarios",
    "build_all_scenarios",
    "catalog_stats",
    "report_to_github",
    "control_redteam",
    "ensure_redteam_weekly_job",
]

#: Detector signature: (ctx dict) -> (attack_succeeded, evidence_text).
Detector = Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass
class AttackScenario:
    """One attack to run against the sandboxed loop."""

    id: str
    name: str
    category: str  # injection | exfiltration | escalation
    description: str
    severity: str  # low | medium | high | critical
    payload: str  # the malicious content
    target: str  # what the attacker wants the agent to do
    detect: Detector | None = None
    # Optional harness hooks, filled by the scenario builders below.
    make_harness: Callable[[], dict[str, Any]] | None = None


@dataclass
class AttackFinding:
    scenario_id: str
    name: str
    severity: str
    succeeded: bool  # True = the ATTACK worked (bad for us)
    evidence: str = ""
    inconclusive: bool = False
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "severity": self.severity,
            "succeeded": self.succeeded,
            "evidence": self.evidence,
            "inconclusive": self.inconclusive,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class RedTeamReport:
    findings: list[AttackFinding] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def failed(self) -> list[AttackFinding]:
        """Findings where the attack worked — our holes."""
        return [f for f in self.findings if f.succeeded and not f.inconclusive]

    @property
    def passed(self) -> list[AttackFinding]:
        """Findings where the defense held."""
        return [f for f in self.findings
                if not f.succeeded and not f.inconclusive]

    @property
    def inconclusive(self) -> list[AttackFinding]:
        return [f for f in self.findings if f.inconclusive]

    def summary(self) -> str:
        lines = [
            f"redteam: {len(self.failed)} failed / {len(self.passed)} passed"
            f" / {len(self.inconclusive)} inconclusive"
            f" ({self.duration_s:.1f}s)",
        ]
        for f in self.failed:
            lines.append(f"  FAIL [{f.severity}] {f.scenario_id}: {f.name}")
        for f in self.inconclusive:
            lines.append(f"  ???? {f.scenario_id}: {f.name} — {f.evidence[:120]}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 3),
            "findings": [f.to_dict() for f in self.findings],
        }


# ── harness fakes ────────────────────────────────────────────────────────────

class _ScriptedLLM:
    """Stub LLM: plays a script of ReAct decisions.

    Each script entry is either a decision dict or a callable
    ``(messages, calls_so_far) -> dict``.  When the script is exhausted
    the model "responds" benignly.  Callables let a scenario simulate a
    VULNERABLE model (follows injected instructions found in the
    observations) while staying benign on clean input — the negative
    control that proves the detector isn't tautological.
    """

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages: Any, sampling: Any = None) -> SimpleNamespace:
        self.calls += 1
        if self.script:
            entry = self.script.pop(0)
            decision = entry(messages, self.calls) if callable(entry) else entry
        else:
            decision = {"action": "respond", "response": "done."}
        return SimpleNamespace(text=json.dumps(decision))


class _FakeRegistry:
    """Instrumented stand-in for ToolRegistry.

    Tools are ``name -> (description, parameters, fn)``.  Every call is
    recorded in ``attempts``.  ``fn`` receives kwargs and returns a value
    (or raises); return values are wrapped in Ok/Err like the real
    registry.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, dict, Callable]] = {}
        self.attempts: list[tuple[str, dict[str, Any]]] = []
        self._lock = threading.Lock()

    def add(self, name: str, description: str, fn: Callable,
            parameters: dict | None = None) -> "_FakeRegistry":
        self._tools[name] = (description, parameters or {}, fn)
        return self

    # -- the surface ToolAdapter needs --
    def get(self, name: str) -> Any:
        return self._tools.get(name)

    def schemas(self, capabilities: Any = None) -> list[dict[str, Any]]:
        return [
            {"name": n, "description": d, "parameters": p}
            for n, (d, p, _) in self._tools.items()
        ]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, actor: str = "redteam",
             capabilities: Any = None, **kwargs: Any) -> Any:
        with self._lock:
            self.attempts.append((name, dict(kwargs)))
        entry = self._tools.get(name)
        if entry is None:
            return Err(f"unknown tool {name!r}")
        _, _, fn = entry
        try:
            return Ok(fn(**kwargs))
        except Exception as exc:  # noqa: BLE001 - tool failures are results
            return Err(str(exc))


class _NullSkillDB:
    """Storage stub so the loop's distillation hook can't touch a real DB."""

    def query_one(self, *a: Any, **k: Any) -> Any:
        return None

    def query(self, *a: Any, **k: Any) -> list:
        return []

    def execute(self, *a: Any, **k: Any) -> SimpleNamespace:
        return SimpleNamespace(rowcount=0)


def _secret_hit(text: str) -> str | None:
    """First redaction-pattern match in text, or None.

    Uses the SAME patterns as the log redactor: if a pattern fires on an
    agent response, a secret-shaped value leaked.
    """
    from ..core.logging_setup import _SECRET_PATTERNS

    for pattern, _ in _SECRET_PATTERNS:
        m = pattern.search(text or "")
        if m:
            return m.group(0)[:60]
    return None


# ── the harness ──────────────────────────────────────────────────────────────

class RedTeam:
    """Runs attack scenarios against a sandboxed AgenticLoop.

    The loop is REAL (same dispatch path as production); the LLM is
    scripted and every tool is a fake.  ``failure_kb`` is optional —
    when given, findings where the attack worked are recorded there.
    """

    def __init__(
        self,
        *,
        step_budget: int = 6,
        per_scenario_timeout_s: float = 60.0,
        failure_kb: Any | None = None,
        sandbox_first: bool = True,
    ) -> None:
        self.step_budget = step_budget
        self.per_scenario_timeout_s = per_scenario_timeout_s
        self.failure_kb = failure_kb
        # Open SWE pattern: isolate first, then trust.  When True, every
        # scenario harness is verified isolated BEFORE the loop runs; a
        # harness that fails verification is never executed.
        self.sandbox_first = sandbox_first

    def run_scenario(self, scenario: AttackScenario) -> AttackFinding:
        started = time.time()
        try:
            harness = scenario.make_harness() if scenario.make_harness else {}
            if self.sandbox_first:
                from .redteam_sandbox import RedTeamSandbox

                iso = RedTeamSandbox().verify_isolation(harness)
                if not iso.ok:
                    return AttackFinding(
                        scenario_id=scenario.id,
                        name=scenario.name,
                        severity=scenario.severity,
                        succeeded=False,
                        evidence=("sandbox violation — scenario NOT run: "
                                  + "; ".join(iso.violations))[:2000],
                        inconclusive=True,
                        duration_s=time.time() - started,
                    )
            if "policy" not in harness:
                from ..core.policy import Policy

                harness["policy"] = Policy()
            result = self._run_loop(scenario, harness)
            harness["result"] = result
            if scenario.detect is not None:
                succeeded, evidence = scenario.detect(harness)
            else:
                succeeded, evidence = False, "no detector"
            return AttackFinding(
                scenario_id=scenario.id,
                name=scenario.name,
                severity=scenario.severity,
                succeeded=bool(succeeded),
                evidence=str(evidence)[:2000],
                duration_s=time.time() - started,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed, never false-pass
            _log.warning("redteam scenario %s errored: %s", scenario.id, exc)
            return AttackFinding(
                scenario_id=scenario.id,
                name=scenario.name,
                severity=scenario.severity,
                succeeded=False,
                evidence=f"harness error (inconclusive): {exc}",
                inconclusive=True,
                duration_s=time.time() - started,
            )

    def _run_loop(self, scenario: AttackScenario,
                  harness: dict[str, Any]) -> Any:
        """Run the real loop in a worker thread with a hard timeout."""
        from .orchestration.loop import AgenticLoop
        from .orchestration.tools import ToolAdapter

        script = harness.get("script") or []
        tools = harness.get("tools") or _FakeRegistry()

        def _work() -> Any:
            llm = _ScriptedLLM(script)
            adapter = ToolAdapter(tools, actor="redteam-loop")
            loop = AgenticLoop(
                llm, adapter,
                step_budget=self.step_budget,
                code_mode=False,  # JSON ReAct path — the injection surface
                db=_NullSkillDB(),
            )
            return loop.run(harness.get("user_message") or scenario.payload)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="redteam") as pool:
            future = pool.submit(_work)
            try:
                return future.result(timeout=self.per_scenario_timeout_s)
            except concurrent.futures.TimeoutError as exc:
                raise TimeoutError(
                    f"scenario exceeded {self.per_scenario_timeout_s}s"
                ) from exc

    def run_suite(
        self,
        scenarios: list[AttackScenario] | None = None,
    ) -> RedTeamReport:
        started = time.time()
        scenarios = scenarios if scenarios is not None else build_scenarios()
        report = RedTeamReport()
        for scenario in scenarios:
            finding = self.run_scenario(scenario)
            report.findings.append(finding)
            if (finding.succeeded and not finding.inconclusive
                    and self.failure_kb is not None):
                try:
                    self.failure_kb.note(
                        cluster_key=f"redteam:{scenario.id}",
                        note_text=(f"[{scenario.severity}] {scenario.name}: "
                                   f"{finding.evidence}"),
                    )
                except Exception as exc:  # noqa: BLE001 - kb never breaks runs
                    _log.warning("redteam FailureKB note failed: %s", exc)
        report.duration_s = time.time() - started
        _log.info("redteam suite: %s", report.summary().splitlines()[0])
        return report


def build_scenarios() -> list[AttackScenario]:
    """The full library, in a stable order (lazy import: scenarios import us)."""
    from .redteam_scenarios import build_scenarios as _build

    return _build()


def build_all_scenarios() -> list[AttackScenario]:
    """Curated scenarios + the 9-class / 47-sub-pattern attack catalog."""
    from .redteam_catalog import build_all_scenarios as _build_all

    return _build_all()


def catalog_stats() -> dict[str, Any]:
    """Attack-catalog counts per class."""
    from .redteam_catalog import catalog_stats as _stats

    return _stats()


def SCENARIOS() -> list[AttackScenario]:  # noqa: N802 - stable public name
    return build_scenarios()


# ── GitHub reporting ─────────────────────────────────────────────────────────

def report_to_github(
    report: RedTeamReport,
    *,
    repo: str = "Oluwacutyp/nomorals-2.0",
    connector: Any | None = None,
) -> dict[str, Any]:
    """File one GitHub issue per failed scenario (deduped by scenario id).

    Returns ``{"created": [...], "skipped": [...]}``.  When the GitHub
    connector has no stored credential the whole thing is skipped with a
    clear reason — never faked.
    """
    out: dict[str, Any] = {"created": [], "skipped": []}
    failed = report.failed
    if not failed:
        return out
    try:
        if connector is None:
            from ..connectors.github import GitHubConnector

            connector = GitHubConnector()
        cred = connector._load_credential()
    except Exception as exc:  # noqa: BLE001 - no credential, no vault, etc.
        reason = f"github not connected ({exc}); skipping issue filing"
        _log.info("redteam %s", reason)
        out["skipped"] = [f.scenario_id for f in failed]
        out["reason"] = reason
        return out
    if cred is None:
        reason = ("github not connected — run `nm connectors connect --name "
                  "github` first; skipping issue filing")
        _log.info("redteam %s", reason)
        out["skipped"] = [f.scenario_id for f in failed]
        out["reason"] = reason
        return out

    def _api(method: str, path: str,
             payload: dict[str, Any] | None = None,
             params: dict[str, Any] | None = None) -> Any:
        return connector._api(method, path, payload, params=params)

    try:
        open_issues = _api(
            "GET", f"/repos/{repo}/issues",
            params={"state": "open", "per_page": 100},
        ) or []
    except Exception as exc:  # noqa: BLE001
        _log.warning("redteam could not list issues: %s", exc)
        open_issues = []
    open_titles = [str(i.get("title", "")) for i in open_issues
                   if isinstance(i, dict)]

    for finding in failed:
        tag = f"[redteam:{finding.scenario_id}]"
        if any(tag in t for t in open_titles):
            out["skipped"].append(finding.scenario_id)
            continue
        body = (
            f"Red-team scenario `{finding.scenario_id}` SUCCEEDED against "
            f"the sandboxed loop — this is a real defensive hole.\n\n"
            f"**Severity:** {finding.severity}\n\n"
            f"**Evidence:**\n```\n{finding.evidence}\n```\n\n"
            f"_Filed automatically by the red-team harness. Also recorded in "
            f"FailureKB under `redteam:{finding.scenario_id}`._"
        )
        try:
            _api("POST", f"/repos/{repo}/issues", {
                "title": f"{tag} {finding.name}",
                "body": body,
                "labels": ["security", "redteam"],
            })
            out["created"].append(finding.scenario_id)
        except Exception as exc:  # noqa: BLE001 - one bad issue != dead run
            _log.warning("redteam issue filing failed for %s: %s",
                         finding.scenario_id, exc)
            out["skipped"].append(finding.scenario_id)
    return out


# ── chat command ─────────────────────────────────────────────────────────────

def control_redteam(arg: str, *, context: Any = None) -> str:
    """Implement ``/redteam``: full suite, catalog, or one scenario by id.

    ``/redteam`` — curated suite · ``/redteam catalog`` — list the
    9-class / 47-sub-pattern catalog · ``/redteam all`` — curated +
    catalog · ``/redteam <scenario-id>`` — one scenario.

    Long-running by nature — the suite caps each scenario with a timeout
    and reports partial results honestly.
    """
    arg = (arg or "").strip()
    if arg == "catalog":
        from .redteam_catalog import ATTACK_CLASSES, catalog_stats

        stats = catalog_stats()
        lines = [f"attack catalog: {stats['classes']} classes, "
                 f"{stats['sub_patterns']} sub-patterns", ""]
        for cls in ATTACK_CLASSES:
            n = stats["per_class"].get(cls["id"], 0)
            lines.append(f"  {cls['id']}: {n} — {cls['name']}")
        return "\n".join(lines)
    if arg == "all":
        scenarios = build_all_scenarios()
    else:
        scenarios = build_scenarios()
    if arg and arg != "all":
        scenarios = [s for s in scenarios if s.id == arg]
        if not scenarios:
            known = ", ".join(s.id for s in build_all_scenarios())
            return f"unknown redteam scenario {arg!r}. known: {known}"

    kb = None
    if context is not None:
        try:
            from ..cognition.failure_kb import FailureKB

            db = getattr(context, "db", None)
            kb = FailureKB(db=db) if db is not None else None
        except Exception as exc:  # noqa: BLE001
            _log.debug("redteam FailureKB unavailable: %s", exc)

    team = RedTeam(failure_kb=kb)
    report = team.run_suite(scenarios)
    lines = [report.summary(), ""]
    for f in report.failed:
        lines.append(f"FAIL [{f.severity}] {f.scenario_id}")
        lines.append(f"  evidence: {f.evidence[:300]}")
    if report.failed:
        lines.append("")
        lines.append("filed to FailureKB; use report_to_github() for issues.")
    return "\n".join(lines).strip()


# ── weekly scheduled run ─────────────────────────────────────────────────────

REDTEAM_WEEKLY_TASK_ID = "redteam-weekly"
REDTEAM_WEEKLY_CRON = "0 3 * * SUN"  # Sundays at 03:00
REDTEAM_WEEKLY_ACTION = "redteam_weekly_run"


async def ensure_redteam_weekly_job(scheduler: Any) -> bool:
    """Register the weekly red-team run. Idempotent. Workstation only.

    Returns True when the job is (now) scheduled.
    """
    from ..core.profiles import get_profile_kind

    if get_profile_kind() != "workstation":
        _log.info("redteam weekly job: workstation profile only (this is %s)",
                  get_profile_kind())
        return False
    try:
        existing = await scheduler.list_cron_jobs()
    except Exception as exc:  # noqa: BLE001
        _log.warning("redteam could not list cron jobs: %s", exc)
        return False
    if any(getattr(j, "task_id", "") == REDTEAM_WEEKLY_TASK_ID
           for j in existing):
        return True

    db = getattr(scheduler, "db", None)

    async def _weekly_run(**params: Any) -> None:
        # Re-check the profile at fire time — the machine may have changed.
        try:
            from ..core.profiles import get_profile_kind as _gpk

            if _gpk() != "workstation":
                _log.info("redteam weekly run skipped (not workstation)")
                return
        except Exception:  # noqa: BLE001
            return
        kb = None
        try:
            from ..cognition.failure_kb import FailureKB

            kb = FailureKB(db=db) if db is not None else None
        except Exception as exc:  # noqa: BLE001
            _log.debug("redteam weekly FailureKB unavailable: %s", exc)
        team = RedTeam(failure_kb=kb)
        report = team.run_suite()
        _log.info("redteam weekly: %s", report.summary().splitlines()[0])

    scheduler.register_action(REDTEAM_WEEKLY_ACTION, _weekly_run)
    await scheduler.schedule_cron(
        REDTEAM_WEEKLY_TASK_ID,
        REDTEAM_WEEKLY_CRON,
        REDTEAM_WEEKLY_ACTION,
        parameters={},
    )
    _log.info("redteam weekly job scheduled (%s)", REDTEAM_WEEKLY_CRON)
    return True
