"""Execute skill packages through the real tool dispatch.

:class:`SkillRunner` takes a skill's ordered tool chain and runs each
step via :meth:`nomorals.tools.registry.ToolRegistry.call` — the same
capability-gated, audited dispatch every agent uses.  There is no
parallel dispatch path: a skill step *is* a tool call.

Data threading: a step with no wiring entry declared receives the run
input unchanged; a step with wiring receives exactly its resolved
mapping (``$input.<path>``, ``$<n>.<path>``, ``$<step_id>.<path>``,
``$last.<path>``, ``$env.VARNAME`` — see ``nomorals/skills/manifest.py``).
No silent merges.

Execution policy (from the manifest, all optional):
  * ``retries`` — Temporal-style retry with exponential backoff.
    Permanent failures (wiring, validation, capability denial, unknown
    tool) surface immediately; transient ones retry up to max_attempts.
  * ``timeout_s`` — per-step wall-clock ceiling.  A hung tool fails the
    step instead of hanging the run.
  * ``on_error`` — GitHub Actions ``continue-on-error``: a failed step
    is recorded but the chain continues with a fallback value.

Every run records a bench entry (success + latency) and, on failure,
persists a structured :class:`nomorals.skills.repair.RepairTicket` with
a concrete suggested fix.  Both stores are optional constructor
dependencies so the runner stays testable without a database.

``run(..., dry_run=True)`` validates the manifest, applies defaults and
coercion, resolves every wiring expression it can, and returns the
planned step/parameter table — invoking nothing (Airflow ``--dry-run``
/ flowdular execution-mode gold).  Dry runs never touch bench or
tickets.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..core.result import Err, Ok
from .manifest import WiringError, parse_wiring_root
from .repair import RepairTicket, build_ticket

if TYPE_CHECKING:  # pragma: no cover
    from ..core.policy import CapabilitySet
    from ..tools.registry import ToolRegistry
    from .bench import SkillBench
    from .registry import InstalledSkill, SkillRegistry
    from .repair import RepairTicketStore

_log = get_logger(__name__)

__all__ = ["SkillRunner", "SkillResult", "StepResult", "format_result",
           "DRY_RUN_SKIPPED"]

#: What a dry run explicitly does not do — the dry-run contract receipt.
DRY_RUN_SKIPPED = ("tool execution", "bench recording",
                   "repair ticket filing")

#: Error-message markers that mark a failure PERMANENT — retried never.
#: (Temporal rule #1: classify errors; permanent failures surface.)
_PERMANENT_MARKERS = (
    "could not resolve wiring",
    "wiring",
    "invalid",
    "validation",
    "missing required",
    "unknown tool",
    "not registered",
    "capability",
    "denied",
    "forbidden",
    "disabled",
    "bad wiring expression",
)


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
    attempts: int = 1
    continued: bool = False  # failed, but on_error.continue carried the chain
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "ok": self.ok,
            "input": dict(self.input),
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
            "attempts": self.attempts,
            "continued": self.continued,
            "dry_run": self.dry_run,
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
    run_id: str = ""
    dry_run: bool = False
    continued_steps: list[int] = field(default_factory=list)

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
            "run_id": self.run_id,
            "dry_run": self.dry_run,
            "continued_steps": list(self.continued_steps),
        }


def format_result(result: SkillResult, *, verbose: bool = False) -> str:
    """Render a skill run as a readable step timeline.

    The god-tier bar for a CLI-driven agent OS: what ran, how long each
    step took, where it broke, and which repair ticket to read next.
    Plain text, no dependencies.
    """
    status = "DRY RUN" if result.dry_run else ("✓ OK" if result.ok
                                               else "✗ FAILED")
    head = (f"skill {result.name} v{result.version} — {status} "
            f"({result.latency_ms:.1f}ms, run {result.run_id or '—'})")
    lines = [head]
    if result.dry_run:
        lines.append("  planned actions: " +
                     ", ".join(f"step {s.index} {s.tool!r}"
                               for s in result.steps))
        lines.append("  checked: input schema, wiring references, "
                     "retry/timeout policy")
        lines.append("  skipped: " + ", ".join(DRY_RUN_SKIPPED))
        lines.append("  status: " + ("ready" if result.ok else "blocked"))
    for step in result.steps:
        if step.dry_run:
            mark = "·"
        elif step.ok:
            mark = "✓"
        elif step.continued:
            mark = "~"
        else:
            mark = "✗"
        detail = f"{step.latency_ms:.1f}ms"
        if step.attempts > 1:
            detail += f", {step.attempts} attempts"
        if step.continued:
            detail += ", continued on error"
        line = f"  {mark} step {step.index} {step.tool!r}  {detail}"
        if verbose:
            line += f"\n      in:  {step.input}"
            if step.ok or step.continued:
                out = repr(step.output)
                line += f"\n      out: {out[:160]}"
        if step.error and not step.ok:
            line += f"\n      ✗ {step.error[:220]}"
        lines.append(line)
    if result.continued_steps:
        lines.append(f"  ~ continued past failed step(s): "
                     f"{', '.join(map(str, result.continued_steps))}")
    if not result.ok and result.error:
        lines.append(f"  ✗ {result.error[:240]}")
    if result.ticket is not None:
        first_fix = (result.ticket.suggested_fix or "").split(". ")[0]
        lines.append(f"  → repair ticket {result.ticket.id}: {first_fix}")
    return "\n".join(lines)


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
        on_step: Callable[[StepResult], None] | None = None,
        allow_env: bool = False,
    ) -> None:
        self.registry = registry
        self.tools = tools
        self.bench = bench
        self.tickets = tickets
        self.actor = actor
        self.capabilities = capabilities
        self.on_step = on_step
        self.allow_env = allow_env

    # ── main entry ──────────────────────────────────────────────────────
    def run(self, name: str, input_data: dict[str, Any] | None = None, *,
            version: str | None = None,
            dry_run: bool = False) -> SkillResult:
        """Execute a skill's tool chain.  Never raises for skill-level
        problems (unknown/disabled skill, bad input) — those come back as
        ``ok=False`` results with a repair ticket.

        ``dry_run=True`` plans the run without invoking any tool: the
        manifest is validated, defaults/coercion applied, wiring resolved
        as far as possible, and the planned steps returned.  No bench
        entries, no tickets.
        """
        started = time.perf_counter()
        run_id = new_short_id("sr")
        skill_input = dict(input_data or {})

        installed = self.registry.get(name, version)
        if installed is None:
            return self._finish(
                name, version or "", started, run_id, ok=False,
                failed_step=-2,
                error=f"unknown skill {name!r}",
                unknown_skill=True)
        if not installed.enabled:
            return self._finish(
                installed.name, installed.version, started, run_id,
                ok=False, failed_step=-2,
                error=f"skill {installed.name!r} is disabled",
                disabled=True)

        manifest = installed.manifest
        if not manifest.tools:
            return self._finish(
                installed.name, installed.version, started, run_id,
                ok=False, failed_step=-2,
                error=f"skill {installed.name!r} has an empty tool chain")

        # Declared defaults were parsed but never applied — fill them
        # before validation so optional-with-default keys validate.
        skill_input = manifest.apply_defaults(skill_input)
        if manifest.coerce_inputs:
            skill_input, _notes = manifest.coerce_input(skill_input)

        input_problems = manifest.validate_input(skill_input)
        if input_problems:
            return self._finish(
                installed.name, installed.version, started, run_id,
                ok=False, failed_step=-1,
                error="; ".join(input_problems),
                step_input=skill_input)

        if dry_run:
            return self._dry_run(installed, skill_input, started, run_id)

        steps: list[StepResult] = []
        step_outputs: list[Any] = []
        outputs: dict[str, Any] = {}
        continued: list[int] = []
        for index, tool_name in enumerate(manifest.tools):
            step = self._run_step(installed, index, tool_name, skill_input,
                                  step_outputs)
            steps.append(step)
            self._emit_step(step)
            if step.continued:
                continued.append(index)
                step_outputs.append(step.output)
                outputs[str(index)] = step.output
                continue
            if not step.ok:
                return self._finish(
                    installed.name, installed.version, started, run_id,
                    ok=False, failed_step=index, error=step.error,
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
                installed.name, installed.version, started, run_id,
                ok=False, failed_step=len(steps) - 1,
                error="; ".join(output_problems),
                steps=steps, outputs=outputs,
                step_input=steps[-1].input if steps else {},
                tool=manifest.tools[-1] if manifest.tools else "",
                step_index=len(steps) - 1,
                missing_keys=_missing_key_names(manifest, final_output))

        result = SkillResult(
            ok=True, name=installed.name, version=installed.version,
            steps=steps, outputs=outputs,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            run_id=run_id, continued_steps=continued)
        self._record_bench(installed, result)
        return result

    # ── dry run: plan without invoking ───────────────────────────────────
    def _dry_run(self, installed: "InstalledSkill",
                 skill_input: dict[str, Any], started: float,
                 run_id: str) -> SkillResult:
        """Validate + resolve everything possible; invoke nothing."""
        manifest = installed.manifest
        steps: list[StepResult] = []
        issues: list[str] = []
        for index, tool_name in enumerate(manifest.tools):
            planned, step_issues = self._dry_plan_step(
                manifest, index, tool_name, skill_input)
            issues.extend(step_issues)
            steps.append(planned)
            self._emit_step(planned)
        ok = not issues
        error = "; ".join(issues) if issues else ""
        return SkillResult(
            ok=ok, name=installed.name, version=installed.version,
            steps=steps, outputs={},
            latency_ms=(time.perf_counter() - started) * 1000.0,
            failed_step=None if ok else -3, error=error,
            run_id=run_id, dry_run=True)

    def _dry_plan_step(self, manifest: Any, index: int, tool_name: str,
                       skill_input: dict[str, Any]
                       ) -> tuple[StepResult, list[str]]:
        """Resolve one step's params against the input shape only.

        ``$input``/``$env`` refs resolve for real; step-output refs are
        *deferred* (their values don't exist yet) after checking the
        reference itself is legal — a forward reference is an issue, not
        a deferral.
        """
        issues: list[str] = []
        wiring = manifest.wiring[index] \
            if 0 <= index < len(manifest.wiring) else {}
        params: dict[str, Any] = {}
        if not wiring:
            params = dict(skill_input)
        else:
            for param_name, expr in wiring.items():
                value, issue = self._dry_resolve_param(
                    manifest, expr, index, skill_input)
                params[param_name] = value
                if issue:
                    issues.append(f"step {index} ({tool_name}) param "
                                  f"{param_name!r}: {issue}")
        step = StepResult(index=index, tool=tool_name, ok=not issues,
                          input=params, error="; ".join(issues),
                          dry_run=True)
        return step, issues

    def _dry_resolve_param(self, manifest: Any, expr: Any, index: int,
                           skill_input: dict[str, Any]
                           ) -> tuple[Any, str | None]:
        root, path = parse_wiring_root(expr)
        if root is None and path is None:
            return expr, None  # literal
        if root is None:
            return f"<bad: {expr}>", path or "malformed expression"
        if root == "input":
            try:
                value = parse_resolve(expr, skill_input, index)
                return value, None
            except WiringError as exc:
                return f"<unresolvable: {expr}>", str(exc)
        if root == "env":
            if not self.allow_env:
                return (f"<unresolvable: {expr}>",
                        "$env references are disabled (allow_env=False)")
            import os as _os
            var = (path or "")[1:]
            if var in _os.environ:
                return f"<env:{var}>", None
            return (f"<unresolvable: {expr}>",
                    f"environment variable {var!r} is not set")
        # Step-output refs: validate legality, then defer.
        ids = manifest.step_ids or []
        if root == "last":
            if index == 0:
                return f"<bad: {expr}>", "$last used at step 0"
            return f"<deferred: {expr}>", None
        if root.isdigit():
            if int(root) >= index:
                return (f"<bad: {expr}>",
                        f"${root} points at a step that has not run yet")
            return f"<deferred: {expr}>", None
        if root not in ids:
            return f"<bad: {expr}>", f"unknown step id {root!r}"
        if ids.index(root) >= index:
            return (f"<bad: {expr}>",
                    f"step id {root!r} has not run yet")
        return f"<deferred: {expr}>", None

    # ── one step: retry + timeout + continue-on-error ────────────────────
    def _run_step(self, installed: "InstalledSkill", index: int,
                  tool_name: str, skill_input: dict[str, Any],
                  step_outputs: list[Any]) -> StepResult:
        manifest = installed.manifest
        step_started = time.perf_counter()
        try:
            kwargs = manifest.step_input(index, skill_input, step_outputs,
                                         allow_env=self.allow_env)
        except WiringError as exc:
            return StepResult(
                index=index, tool=tool_name, ok=False, input={},
                error=f"could not resolve wiring: {exc}",
                latency_ms=(time.perf_counter() - step_started) * 1000.0)

        policy = manifest.retry_policy()
        max_attempts = policy["max_attempts"] if policy else 1
        timeout_s = float(manifest.timeout_s or 0.0)
        attempts = 0
        last_error = ""
        while attempts < max_attempts:
            attempts += 1
            outcome, error = self._dispatch_once(
                installed.name, tool_name, kwargs, timeout_s)
            if outcome is not None:
                latency_ms = (time.perf_counter() - step_started) * 1000.0
                return StepResult(index=index, tool=tool_name, ok=True,
                                  input=kwargs, output=outcome,
                                  latency_ms=latency_ms,
                                  attempts=attempts)
            last_error = error
            # Temporal rule #1: permanent failures surface immediately.
            if attempts < max_attempts and policy is not None \
                    and self._should_retry(error, policy):
                delay = min(policy["initial_backoff_s"]
                            * (policy["backoff_multiplier"] ** (attempts - 1)),
                            policy["max_backoff_s"])
                _log.info("skill %s step %d attempt %d failed (%s); "
                          "retrying in %.2fs", installed.name, index,
                          attempts, error[:80], delay)
                time.sleep(max(0.0, delay))
                continue
            break

        latency_ms = (time.perf_counter() - step_started) * 1000.0
        # GitHub Actions continue-on-error: record the failure, carry the
        # chain with the fallback as this step's effective output.
        on_error = manifest.on_error_for(index)
        if on_error.get("continue"):
            fallback = self._resolve_fallback(
                manifest, on_error.get("fallback"), skill_input,
                step_outputs, index)
            _log.info("skill %s step %d failed but on_error.continue is "
                      "set — carrying on with fallback", installed.name,
                      index)
            return StepResult(index=index, tool=tool_name, ok=False,
                              input=kwargs, output=fallback,
                              error=last_error, latency_ms=latency_ms,
                              attempts=attempts, continued=True)
        return StepResult(index=index, tool=tool_name, ok=False,
                          input=kwargs, error=last_error,
                          latency_ms=latency_ms, attempts=attempts)

    def _dispatch_once(self, skill_name: str, tool_name: str,
                       kwargs: dict[str, Any],
                       timeout_s: float) -> tuple[Any | None, str]:
        """One tool call.  Returns ``(value, "")`` on success or
        ``(None, message)`` on failure.  Never raises."""
        if timeout_s > 0:
            return self._dispatch_with_timeout(skill_name, tool_name,
                                               kwargs, timeout_s)
        try:
            outcome = self.tools.call(
                tool_name, actor=f"{self.actor}:{skill_name}",
                capabilities=self.capabilities, **kwargs)
        except Exception as exc:  # noqa: BLE001 — dispatch must not raise
            return None, f"tool dispatch raised: {exc}"
        return self._interpret_outcome(tool_name, outcome)

    def _dispatch_with_timeout(self, skill_name: str, tool_name: str,
                               kwargs: dict[str, Any],
                               timeout_s: float) -> tuple[Any | None, str]:
        """Run the tool call in a daemon thread; fail on timeout.

        The thread is daemonized — a hung tool cannot hang the run, and
        it cannot hang process exit either.  (Temporal rule #2: always
        set a maximum.)
        """
        box: dict[str, Any] = {}

        def _target() -> None:
            try:
                box["outcome"] = self.tools.call(
                    tool_name,
                    actor=f"{self.actor}:{skill_name}",
                    capabilities=self.capabilities, **kwargs)
            except Exception as exc:  # noqa: BLE001
                box["raised"] = exc

        thread = threading.Thread(target=_target, daemon=True,
                                  name=f"skill-step-{tool_name}")
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            return None, (f"step timed out after {timeout_s:g}s waiting "
                          f"for tool {tool_name!r}")
        if "raised" in box:
            return None, f"tool dispatch raised: {box['raised']}"
        return self._interpret_outcome(tool_name, box.get("outcome"))

    @staticmethod
    def _interpret_outcome(tool_name: str,
                           outcome: Any) -> tuple[Any | None, str]:
        if isinstance(outcome, Err):
            error = outcome.error
            message = getattr(error, "message", None) or str(error)
            return None, message
        if isinstance(outcome, Ok):
            return outcome.value, ""
        # Defensive: an unknown Outcome subtype is a failure, not a crash.
        return None, (f"unexpected outcome type "
                      f"{type(outcome).__name__} from tool {tool_name!r}")

    @staticmethod
    def _should_retry(error: str, policy: dict[str, Any]) -> bool:
        """Temporal-style classification: retry unless permanent."""
        low = (error or "").lower()
        for marker in policy.get("non_retryable_errors", []):
            if marker and marker in low:
                return False
        for marker in policy.get("retryable_errors", []):
            if marker and marker in low:
                return True
        for marker in _PERMANENT_MARKERS:
            if marker in low:
                return False
        return True

    def _resolve_fallback(self, manifest: Any, fallback: Any,
                          skill_input: dict[str, Any],
                          step_outputs: list[Any], step_index: int) -> Any:
        """Resolve a fallback value, deep-resolving ``$`` refs inside
        nested dicts/lists.  An unresolvable ref becomes None rather
        than sinking the continuation."""
        from .manifest import resolve_expression

        def _walk(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("$"):
                try:
                    return resolve_expression(
                        value, skill_input=skill_input,
                        step_outputs=step_outputs, step_index=step_index,
                        step_ids=manifest.step_ids or None,
                        allow_env=self.allow_env)
                except WiringError as exc:
                    _log.debug("on_error fallback ref did not resolve: %s",
                               exc)
                    return None
            if isinstance(value, dict):
                return {k: _walk(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_walk(v) for v in value]
            return value

        return _walk(fallback)

    def _emit_step(self, step: StepResult) -> None:
        if self.on_step is not None:
            try:
                self.on_step(step)
            except Exception as exc:  # noqa: BLE001 — hooks never sink runs
                _log.debug("on_step hook raised: %s", exc)

    # ── finishing: bench + ticket ───────────────────────────────────────
    def _finish(self, name: str, version: str, started: float, run_id: str,
                *, ok: bool, failed_step: int | None,
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
            failed_step=failed_step, error=error, ticket=ticket,
            run_id=run_id)
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
        record_steps = getattr(self.bench, "record_steps", None)
        if callable(record_steps) and result.steps:
            try:
                record_steps(installed.name, installed.version,
                             result.steps)
            except Exception as exc:  # noqa: BLE001 — bench never sinks runs
                _log.debug("skill bench step recording failed: %s", exc)


def parse_resolve(expr: str, skill_input: dict[str, Any],
                  step_index: int) -> Any:
    """Resolve an ``$input``-rooted expression (dry-run helper)."""
    from .manifest import resolve_expression
    return resolve_expression(expr, skill_input=skill_input,
                              step_outputs=[], step_index=step_index)


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
