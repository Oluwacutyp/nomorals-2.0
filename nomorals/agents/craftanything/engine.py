"""The engine: one dispatch every caller goes through.

Port of upstream ``Engine::call`` / ``call_inner``
(``crates/engine/src/lib.rs``) and the MCP server's consent + budget
rules (``crates/mcp/src/lib.rs``): look the tool up, check availability,
check consent for consent-class effects, run it with a panic guard,
record evidence, journal the call. The UI, the CLI, MCP and the agent
all dispatch through this one function with a ``Caller`` — there is no
second path that skips the checks.

Devon's differences from upstream, named where they occur:

* The engine wraps Devon's live ``ToolRegistry`` (injected, duck-typed
  ``get``/``call``) instead of owning a ``Registry``: the registry
  stays the single source of tools, capability gating and the audit
  trail, and the engine adds the effects + consent layer on top of it.
  A call that passes the engine still passes the registry's own
  capability check — the two gates compose, they do not replace each
  other.
* Availability is Devon's: unknown tool, deprecated tool (warns once,
  still runs — the registry's own rule).
* The effect check is new: a call whose requested effects are not a
  subset of the tool's declared effects is refused with the missing
  effect named, before anything executes. A tool with no declaration
  at all is derived generously (see :func:`effects.derive_effects`)
  and the journal marks ``effects_known: false`` — the check cannot
  constrain what was never declared, and the journal says so honestly.
* Over-limit results are retained as evidence (upstream: responses over
  the size budget become an ``over_limit`` error pointing at retained
  evidence ``evidence://<id>``). Devon keeps the result and returns a
  summary + the id, so the caller can fetch it — the same retention,
  without dropping the work.
* Results are plain dicts (Devon's tool protocol), not a Rust enum:
  ``{"ok": True, "result": ..., ...}`` or ``{"ok": False,
  "error_code": ..., "error": ..., "remediation": ..., "details": ...}``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import effects as _effects
from .caller import Caller, CancelToken, Progress, ProgressFn
from .consent import Consent, ConsentGate
from .specs import action_effects, declared_effects
from .toolsets import Toolset, list_toolsets, toolset_for

__all__ = [
    "Caller", "CancelToken", "Progress", "ProgressFn",
    "CallOptions", "ToolOutput", "CraftEngine",
]


@dataclass
class CallOptions:
    """Options for one call (upstream, verbatim shape)."""

    caller: Caller = Caller.UI
    consent: Consent = Consent.NOT_ASKED
    cancel: CancelToken | None = None
    progress: ProgressFn | None = None

    def with_consent(self, consent: Consent) -> CallOptions:
        return CallOptions(
            caller=self.caller, consent=consent,
            cancel=self.cancel, progress=self.progress)


@dataclass
class ToolOutput:
    """What a tool run returns (upstream: evidence or a plain value)."""

    kind: str  # "evidence" | "value"
    payload: Any
    evidence_id: str | None = None

    def to_json(self) -> Any:
        if self.kind == "evidence":
            return self.payload
        return {"result": self.payload}


class CraftEngine:
    """Dispatch through the live registry with effects + consent checks.

    ``registry`` is any object with ``get(name)`` returning a spec (or
    ``None``) and ``call(name, /, *, actor, **kwargs)`` returning an
    outcome with ``.ok`` / ``.value`` / ``.error`` — Devon's
    ``ToolRegistry``. The engine never imports the registry: it is
    injected, so this organ stays a leaf over ``core``.
    """

    def __init__(self, registry: Any, *,
                 gate: ConsentGate | None = None,
                 max_result_bytes: int = 8 * 1024 * 1024,
                 evidence_dir: str | Path | None = None) -> None:
        self.registry = registry
        self.gate = gate or ConsentGate()
        self.max_result_bytes = max_result_bytes
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self._evidence: dict[str, dict[str, Any]] = {}
        self._journal: list[dict[str, Any]] = []
        self._enabled_toolsets: set[str] = {"core"}

    # ── inspection ────────────────────────────────────────────────────
    def effective_effects(self, tool: str, spec: Any,
                          action: str | None = None) -> tuple[int, bool]:
        """The effects a tool declares, and whether they were declared.

        Precedence: the explicit table in :mod:`specs` (per tool, then
        per action when one is given), else ``ToolSpec.effects`` when
        the tool declared it, else derivation from kind + capability.
        Returns ``(mask, known)`` — ``known`` is False when nothing was
        ever declared and the mask is only derived.
        """
        table = declared_effects(tool)
        if table is not None:
            if action:
                per_action = action_effects(tool, action)
                if per_action is not None:
                    return per_action, True
            return table, True
        declared = getattr(spec, "effects", None)
        if declared is not None:
            return int(declared), True
        return _effects.derive_effects(
            getattr(spec, "kind", "") or "",
            getattr(spec, "capability", "") or "",
            getattr(spec, "annotations", None) or {}), False

    def inspect(self, tool: str) -> dict[str, Any] | None:
        """A tool's spec with its effects, hints and toolset."""
        spec = self.registry.get(tool)
        if spec is None:
            return None
        effects, known = self.effective_effects(tool, spec)
        return {
            "name": tool,
            "title": getattr(spec, "title", "") or tool,
            "description": getattr(spec, "description", ""),
            "capability": getattr(spec, "capability", ""),
            "effects": _effects.effect_names(effects),
            "effects_known": known,
            "hints": {
                "read_only": _effects.read_only_hint(effects),
                "destructive": _effects.destructive_hint(effects),
                "open_world": _effects.open_world_hint(effects),
                "needs_consent": _effects.needs_consent(effects),
            },
            "toolset": toolset_for(tool, spec).id,
            "deprecated": bool(getattr(spec, "deprecated", False)),
        }

    def list_tools(self, *,
                   toolsets: set[str] | frozenset[str] | None = None
                   ) -> list[dict[str, Any]]:
        """Tool summaries for the given toolsets (upstream ``tools/list``).

        ``None`` lists everything; otherwise only the named toolsets —
        agents see Core plus whatever they enabled, which keeps their
        context small. ``core`` is always included: Core is the
        essentials, and an agent without it cannot do anything.
        """
        wanted = None if toolsets is None else set(toolsets) | {"core"}
        out = []
        for name, spec in self.registry._tools.items():
            toolset = toolset_for(name, spec)
            if wanted is not None and toolset.id not in wanted:
                continue
            effects, known = self.effective_effects(name, spec)
            out.append({
                "name": name,
                "title": getattr(spec, "title", "") or name,
                "description": getattr(spec, "description", ""),
                "capability": getattr(spec, "capability", ""),
                "effects": _effects.effect_names(effects),
                "effects_known": known,
                "hints": {
                    "read_only": _effects.read_only_hint(effects),
                    "destructive": _effects.destructive_hint(effects),
                    "open_world": _effects.open_world_hint(effects),
                    "needs_consent": _effects.needs_consent(effects),
                },
                "toolset": toolset.id,
            })
        return sorted(out, key=lambda item: item["name"])

    def enable_toolset(self, toolset_id: str) -> dict[str, Any]:
        """Enable a toolset for this engine's agents (upstream meta-tool)."""
        toolset = Toolset.from_id(toolset_id)
        if toolset is None:
            return {"ok": False, "error_code": "unknown_toolset",
                    "error": f"No toolset named {toolset_id!r}",
                    "remediation": "List toolsets with list_toolsets."}
        self._enabled_toolsets.add(toolset.id)
        return {"ok": True, "enabled": sorted(self._enabled_toolsets)}

    def disable_toolset(self, toolset_id: str) -> dict[str, Any]:
        if toolset_id == "core":
            return {"ok": False, "error_code": "invalid",
                    "error": "The core toolset cannot be disabled.",
                    "remediation": "Core is the essentials every agent needs."}
        self._enabled_toolsets.discard(toolset_id)
        return {"ok": True, "enabled": sorted(self._enabled_toolsets)}

    def toolsets(self) -> list[dict[str, Any]]:
        """Every toolset in use, with counts (upstream ``Registry::toolsets``)."""
        return list_toolsets(self.registry)

    # ── evidence ──────────────────────────────────────────────────────
    def evidence(self, evidence_id: str) -> dict[str, Any] | None:
        """Fetch a retained over-limit result by id (``evidence://<id>``)."""
        record = self._evidence.get(evidence_id)
        if record is not None:
            return record
        if self.evidence_dir is not None:
            path = self.evidence_dir / f"{evidence_id}.json"
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _retain_evidence(self, tool: str, params: dict,
                         result: Any) -> str:
        """Retain an over-limit result; returns its evidence id."""
        blob = json.dumps(result, default=str, sort_keys=True)
        evidence_id = "ev_" + hashlib.sha256(
            f"{tool}:{blob}".encode()).hexdigest()[:16]
        record = {"id": evidence_id, "tool": tool,
                  "params_digest": self._params_digest(params),
                  "result": result, "ts": time.time()}
        self._evidence[evidence_id] = record
        if self.evidence_dir is not None:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            (self.evidence_dir / f"{evidence_id}.json").write_text(
                json.dumps(record, default=str, sort_keys=True),
                encoding="utf-8")
        return evidence_id

    # ── the dispatch ──────────────────────────────────────────────────
    @staticmethod
    def _params_digest(params: dict) -> str:
        blob = json.dumps(params, default=str, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def call(self, tool: str, params: dict | None = None,
             opts: CallOptions | None = None) -> dict[str, Any]:
        """Run a tool by name through every check (upstream ``Engine::call``)."""
        started = time.perf_counter()
        params = dict(params or {})
        opts = opts or CallOptions()
        caller = opts.caller
        progress = opts.progress or (lambda _: None)

        def finish(result: dict[str, Any]) -> dict[str, Any]:
            progress(Progress(1.0, f"{tool} done"))
            self._journal.append({
                "ts": time.time(),
                "tool": tool,
                "caller": caller.value,
                "params_digest": self._params_digest(params),
                "param_keys": sorted(params),
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000, 1),
                "ok": bool(result.get("ok")),
                "error_code": result.get("error_code", ""),
                "consent": result.get("consent", {}),
            })
            return result

        progress(Progress(0.0, f"calling {tool}"))
        if not isinstance(params, dict):
            return finish({"ok": False, "error_code": "bad_input",
                           "error": "params must be an object",
                           "remediation": "Pass tool parameters as an object."})
        if opts.cancel is not None and opts.cancel.is_cancelled():
            return finish({"ok": False, "error_code": "cancelled",
                           "error": f"{tool} was cancelled before it started",
                           "remediation": "Call it again without cancelling."})
        spec = self.registry.get(tool)
        if spec is None:
            return finish({
                "ok": False, "error_code": "bad_input",
                "error": f"No tool named {tool}",
                "remediation": "List tools with list_tools.",
                "details": {"tool": tool}})
        action = params.get("action") if isinstance(
            params.get("action"), str) else None
        effects, known = self.effective_effects(tool, spec, action)
        requested = _effects.infer_requested_effects(action, params)
        per_action = action_effects(tool, action) if action else None
        if per_action is not None:
            requested |= per_action
        missing = requested & ~effects
        if missing:
            names = _effects.effect_names(missing)
            return finish({
                "ok": False, "error_code": "effect_refused",
                "error": (f"{tool} does not declare "
                          f"{', '.join(names)}; this call asks for it"),
                "remediation": ("Declare the effect on the tool's spec, "
                                "or call without the parameter that asks "
                                "for it."),
                "details": {"tool": tool, "missing_effects": names,
                            "declared_effects": _effects.effect_names(effects)}})
        # The consent gate decides on what THIS call will do: when the
        # tool declares per-action effects, the call's effects are the
        # action's plus whatever its parameters ask for; otherwise the
        # tool's declared effects (upstream's rule) decide.
        gate_effects = requested if per_action is not None else effects
        decision = self.gate.decide(tool, gate_effects, caller,
                                    consent=opts.consent)
        consent_info = decision.to_dict()
        if decision.decision == "deny":
            return finish({
                "ok": False, "error_code": "consent_required",
                "error": decision.reason,
                "remediation": ("A standing rule denies this action; "
                                "remove the rule to allow it."),
                "details": {"tool": tool,
                            "effects": _effects.effect_names(gate_effects)},
                "consent": consent_info})
        if decision.decision == "consent_required":
            title = getattr(spec, "title", "") or tool
            return finish({
                "ok": False, "error_code": "consent_required",
                "error": (f"{title} runs code or changes a device, so it "
                          f"needs your OK first."),
                "remediation": ("Approve it when asked, or allow it for "
                                "this tool in standing rules."),
                "details": {"tool": tool,
                            "effects": _effects.effect_names(gate_effects)},
                "consent": consent_info})
        try:
            outcome = self.registry.call(
                tool, actor=caller.value, **params)
        except Exception as exc:  # noqa: BLE001 - the panic guard
            return finish({
                "ok": False, "error_code": "internal",
                "error": f"{tool} hit a bug and stopped safely: {exc}",
                "remediation": "Your work is kept; report the tool name.",
                "details": {"tool": tool},
                "consent": consent_info})
        if not getattr(outcome, "ok", False):
            error = getattr(outcome, "error", None)
            message = getattr(error, "message", None) or str(error or "call failed")
            code = type(error).__name__ if error is not None else "tool_error"
            return finish({
                "ok": False, "error_code": code,
                "error": message,
                "details": {"tool": tool},
                "consent": consent_info})
        result = outcome.value
        blob = json.dumps(result, default=str, sort_keys=True)
        if len(blob.encode("utf-8")) > self.max_result_bytes:
            evidence_id = self._retain_evidence(tool, params, result)
            return finish({
                "ok": True,
                "result": ("Result retained as evidence: too large to "
                           f"return inline ({len(blob)} bytes)."),
                "over_limit": True,
                "evidence_id": evidence_id,
                "evidence_uri": f"evidence://{evidence_id}",
                "effects": _effects.effect_names(effects),
                "effects_known": known,
                "consent": consent_info})
        return finish({
            "ok": True, "result": result,
            "effects": _effects.effect_names(effects),
            "effects_known": known,
            "consent": consent_info})

    # ── journal ───────────────────────────────────────────────────────
    @property
    def journal(self) -> list[dict[str, Any]]:
        return list(self._journal)

    def save_journal(self, path: str | Path) -> Path:
        """Write the call journal + consent journal to one JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(
            {"calls": self._journal, "consent": self.gate.journal},
            default=str, sort_keys=True, indent=2), encoding="utf-8")
        return target
