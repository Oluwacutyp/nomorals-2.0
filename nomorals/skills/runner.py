"""Execute skill packages through the real tool dispatch.

:class:`SkillRunner` takes a skill's ordered tool chain and runs each
step via :meth:`nomorals.tools.registry.ToolRegistry.call` — the same
capability-gated, audited dispatch every agent uses.  There is no
parallel dispatch path: a skill step *is* a tool call.

Data threading: a step with no wiring entry declared receives the run
input unchanged; a step with wiring receives exactly its resolved
mapping (``$input.<path>``, ``$<n>.<path>``, ``$last.<path>`` — see
``nomorals/skills/manifest.py``).  No silent merges.

Every run records a bench entry (success + latency) and, on failure,
persists a structured :class:`nomorals.skills.repair.RepairTicket` with
a concrete suggested fix.  Both stores are optional constructor
dependencies so the runner stays testable without a database.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.logging_setup import get_logger
from ..core.result import Err, Ok
from .manifest import WiringError
from .repair import RepairTicket, build_ticket

if TYPE_CHECKING:  # pragma: no cover
    from ..core.policy import CapabilitySet
    from ..tools.registry import ToolRegistry
    from .bench import SkillBench
    from .registry import InstalledSkill, SkillRegistry
    from .repair import RepairTicketStore

_log = get_logger(__name__)

__all__ = ["SkillRunner", "SkillResult", "StepResult"]


@dataclass
class StepResult:
    """What one tool step did."""

    index: int
    tool: str
    ok: bool
    input: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "ok": self.ok,
            "input": dict(self.input),
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class SkillResult:
    """The outcome of one skill run."""

    ok: bool
    name: str
    version: str
    steps: list[StepResult] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    failed_step: int | None = None
    error: str = ""
    ticket: RepairTicket | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "name": self.name,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "outputs": dict(self.outputs),
            "latency_ms": round(self.latency_ms, 3),
            "failed_step": self.failed_step,
            "error": self.error,
            "ticket": self.ticket.to_dict() if self.ticket else None,
        }


class SkillRunner:
    """Run installed skill packages step by step."""

    def __init__(
        self,
        registry: "SkillRegistry",
        tools: "ToolRegistry",
        *,
        bench: "SkillBench | None" = None,
        tickets: "RepairTicketStore | None" = None,
        actor: str = "skill",
        capabilities: "CapabilitySet | None" = None,
    ) -> None:
        self.registry = registry
        self.tools = tools
        self.bench = bench
        self.tickets = tickets
        self.actor = actor
        self.capabilities = capabilities

    # ── main entry ──────────────────────────────────────────────────────
    def run(self, name: str, input_data: dict[str, Any] | None = None, *,
            version: str | None = None) -> SkillResult:
        """Execute a skill's tool chain.  Never raises for skill-level
        problems (unknown/disabled skill, bad input) — those come back as
        ``ok=False`` results with a repair ticket."""
        started = time.perf_counter()
        skill_input = dict(input_data or {})

        installed = self.registry.get(name, version)
        if installed is None:
            return self._finish(
                name, version or "", started, ok=False, failed_step=-2,
                error=f"unknown skill {name!r}",
                unknown_skill=True)
        if not installed.enabled:
            return self._finish(
                installed.name, installed.version, started, ok=False,
                failed_step=-2,
                error=f"skill {installed.name!r} is disabled",
                disabled=True)

        manifest = installed.manifest
        if not manifest.tools:
            return self._finish(
                installed.name, installed.version, started, ok=False,
                failed_step=-2,
                error=f"skill {installed.name!r} has an empty tool chain")

        input_problems = manifest.validate_input(skill_input)
        if input_problems:
            return self._finish(
                installed.name, installed.version, started, ok=False,
                failed_step=-1,
                error="; ".join(input_problems),
                step_input=skill_input)

        steps: list[StepResult] = []
        step_outputs: list[Any] = []
        outputs: dict[str, Any] = {}
        for index, tool_name in enumerate(manifest.tools):
            step = self._run_step(installed, index, tool_name, skill_input,
                                  step_outputs)
            steps.append(step)
            if not step.ok:
                return self._finish(
                    installed.name, installed.version, started, ok=False,
                    failed_step=index, error=step.error,
                    steps=steps, outputs=outputs,
                    step_input=step.input, tool=tool_name,
                    step_index=index)
            step_outputs.append(step.output)
            outputs[str(index)] = step.output

        # The chain ran; the contract still has to hold on the way out.
        final_output = step_outputs[-1] if step_outputs else None
        output_problems = manifest.validate_output(final_output)
        if output_problems:
            missing = [p for p in output_problems if "missing required" in p]
            return self._finish(
                installed.name, installed.version, started, ok=False,
                failed_step=len(steps) - 1, error="; ".join(output_problems),
                steps=steps, outputs=outputs,
                step_input=steps[-1].input if steps else {},
                tool=manifest.tools[-1] if manifest.tools else "",
                step_index=len(steps) - 1,
                missing_keys=_missing_key_names(manifest, final_output))

        result = SkillResult(
            ok=True, name=installed.name, version=installed.version,
            steps=steps, outputs=outputs,
            latency_ms=(time.perf_counter() - started) * 1000.0)
        self._record_bench(installed, result)
        return result

    # ── one step ────────────────────────────────────────────────────────
    def _run_step(self, installed: "InstalledSkill", index: int,
                  tool_name: str, skill_input: dict[str, Any],
                  step_outputs: list[Any]) -> StepResult:
        manifest = installed.manifest
        step_started = time.perf_counter()
        try:
            kwargs = manifest.step_input(index, skill_input, step_outputs)
        except WiringError as exc:
            return StepResult(
                index=index, tool=tool_name, ok=False, input={},
                error=f"could not resolve wiring: {exc}",
                latency_ms=(time.perf_counter() - step_started) * 1000.0)

        # The real dispatch path: capability-gated, audited, timed.
        outcome = self.tools.call(
            tool_name, actor=f"{self.actor}:{installed.name}",
            capabilities=self.capabilities, **kwargs)
        latency_ms = (time.perf_counter() - step_started) * 1000.0
        if isinstance(outcome, Err):
            error = outcome.error
            message = getattr(error, "message", None) or str(error)
            return StepResult(index=index, tool=tool_name, ok=False,
                              input=kwargs, error=message,
                              latency_ms=latency_ms)
        if isinstance(outcome, Ok):
            return StepResult(index=index, tool=tool_name, ok=True,
                              input=kwargs, output=outcome.value,
                              latency_ms=latency_ms)
        # Defensive: an unknown Outcome subtype is a failure, not a crash.
        return StepResult(index=index, tool=tool_name, ok=False, input=kwargs,
                          error=f"unexpected outcome type "
                                f"{type(outcome).__name__} from tool "
                                f"{tool_name!r}",
                          latency_ms=latency_ms)

    # ── finishing: bench + ticket ───────────────────────────────────────
    def _finish(self, name: str, version: str, started: float, *,
                ok: bool, failed_step: int | None,
                error: str, steps: list[StepResult] | None = None,
                outputs: dict[str, Any] | None = None,
                step_input: dict[str, Any] | None = None,
                tool: str = "", step_index: int = -2,
                missing_keys: list[str] | None = None,
                disabled: bool = False,
                unknown_skill: bool = False) -> SkillResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        ticket: RepairTicket | None = None
        if not ok:
            available = []
            try:
                available = self.tools.names()
            except Exception:  # noqa: BLE001 - ticket must still be built
                available = []
            ticket = build_ticket(
                name, version, step_index=step_index, tool=tool,
                step_input=step_input, error=error,
                missing_keys=missing_keys, available_tools=available,
                disabled=disabled, unknown_skill=unknown_skill)
            if self.tickets is not None:
                try:
                    self.tickets.save(ticket)
                except Exception as exc:  # noqa: BLE001 - never sink a result
                    _log.debug("could not persist repair ticket: %s", exc)
        result = SkillResult(
            ok=ok, name=name, version=version, steps=steps or [],
            outputs=outputs or {}, latency_ms=latency_ms,
            failed_step=failed_step, error=error, ticket=ticket)
        installed = self.registry.get(name, version or None)
        if installed is not None:
            self._record_bench(installed, result)
        elif not unknown_skill:
            # The skill resolved earlier but vanished mid-run (should not
            # happen); still record the bench entry under the given name.
            if self.bench is not None:
                self.bench.record(name, version, success=ok,
                                  latency_ms=latency_ms)
        return result

    def _record_bench(self, installed: "InstalledSkill",
                      result: SkillResult) -> None:
        if self.bench is None:
            return
        self.bench.record(installed.name, installed.version,
                          success=result.ok, latency_ms=result.latency_ms)


def _missing_key_names(manifest: Any, final_output: Any) -> list[str]:
    """Which required output-schema keys are absent from the final output."""
    missing: list[str] = []
    schema = getattr(manifest, "output_schema", {}) or {}
    if not isinstance(final_output, dict):
        return [k for k in schema if isinstance(k, str)]
    for key, spec in schema.items():
        required = True
        if isinstance(spec, str):
            required = not spec.strip().endswith("?")
        elif isinstance(spec, dict):
            required = bool(spec.get("required", True))
            if "default" in spec:
                required = False
        if required and key not in final_output:
            missing.append(key)
    return missing
