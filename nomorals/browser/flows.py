"""Browser flows: a multi-step browsing task as a resumable mission.

A :class:`BrowserFlow` is a list of steps plus persisted progress. Each
step is plain data — ``{kind, target, value, description}`` — and an
*executor* (any callable taking a step dict and returning a result dict)
interprets it. The flow itself never touches the network: it owns the
mission state, the step index, and the JSON checkpoint.

Resumability is the point: after every step the flow persists its index,
results, and status atomically. A fresh :class:`BrowserFlow` loaded from
the same checkpoint with :meth:`load` resumes at exactly the next
unfinished step — a restart never replays completed work, and a step's
result is recorded exactly once per attempt.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.ids import ulid_now

__all__ = [
    "FlowError",
    "FlowStep",
    "BrowserFlow",
    "STEP_KINDS",
]

_log = get_logger(__name__)

#: Step kinds a driver may implement. Unknown kinds fail fast at
#: validation — a flow must never contain a step no driver understands.
STEP_KINDS = frozenset({
    "navigate", "click", "fill", "select", "check", "submit", "extract",
    "wait", "wait_text", "download", "snapshot", "screenshot", "evaluate",
    "scroll", "back", "note",
})


class FlowError(ValueError):
    """A flow is malformed, or an operation is illegal in its state."""


@dataclass
class FlowStep:
    """One step of a flow. ``kind`` names the action; ``target`` is the
    selector/URL/field; ``value`` is the payload (text, timeout, …)."""

    kind: str = ""
    target: str = ""
    value: Any = None
    description: str = ""

    def validate(self) -> list[str]:
        errs: list[str] = []
        kind = (self.kind or "").strip().lower()
        if kind not in STEP_KINDS:
            errs.append(f"unknown step kind {self.kind!r} (known: {sorted(STEP_KINDS)})")
        if kind == "navigate" and not (self.target or "").strip():
            errs.append("navigate step needs a target URL")
        if kind in {"click", "fill", "select", "check", "extract"} and not (self.target or "").strip():
            errs.append(f"{kind} step needs a target selector/field")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FlowStep":
        raw = dict(raw or {})
        return cls(
            kind=str(raw.get("kind") or ""),
            target=str(raw.get("target") or ""),
            value=raw.get("value"),
            description=str(raw.get("description") or ""),
        )


#: Flow lifecycle states.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class BrowserFlow:
    """A resumable multi-step browsing mission.

    ``executor`` is a callable ``step_dict -> result_dict`` supplied by
    the caller (a tab adapter, a daemon client, a test double). The flow
    calls it one step at a time and persists after every step, so
    ``pause()`` / process death / reload all resume cleanly.
    """

    def __init__(
        self,
        steps: list[FlowStep | dict[str, Any]] | None = None,
        *,
        flow_id: str = "",
        name: str = "",
        checkpoint_path: str | Path | None = None,
        executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.flow_id = flow_id or ulid_now()
        self.name = (name or "").strip() or f"flow-{self.flow_id}"
        self.steps: list[FlowStep] = [
            s if isinstance(s, FlowStep) else FlowStep.from_dict(s)
            for s in (steps or [])
        ]
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self.executor = executor
        self.index = 0  # next step to run
        self.status = STATUS_PENDING
        self.results: list[dict[str, Any]] = []
        self.error: str = ""
        self.started_at: float = 0.0
        self.updated_at: float = 0.0
        errs = self.validate()
        if errs:
            raise FlowError("invalid flow: " + "; ".join(errs))

    # -- validation ----------------------------------------------------------
    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.steps:
            errs.append("flow needs at least one step")
        for i, step in enumerate(self.steps):
            for err in step.validate():
                errs.append(f"steps[{i}]: {err}")
        return errs

    @property
    def done(self) -> bool:
        return self.status == STATUS_DONE

    @property
    def next_step(self) -> FlowStep | None:
        if 0 <= self.index < len(self.steps):
            return self.steps[self.index]
        return None

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "flow_id": self.flow_id,
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "index": self.index,
            "status": self.status,
            "results": self.results,
            "error": self.error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Atomic checkpoint write (tmp file + rename)."""
        dest = Path(path).expanduser() if path else self.checkpoint_path
        if dest is None:
            raise FlowError("no checkpoint path — pass one to save() or the constructor")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(dest)
        self.checkpoint_path = dest
        return dest

    @classmethod
    def load(cls, path: str | Path, *,
             executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> "BrowserFlow":
        """Restore a flow from a checkpoint. A loaded flow that was
        ``running`` when it died comes back ``paused`` — it never claims
        to be mid-step, because no step is atomic across a restart."""
        p = Path(path).expanduser()
        if not p.is_file():
            raise FlowError(f"no flow checkpoint at {p}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FlowError(f"corrupt flow checkpoint {p}: {exc}") from exc
        if not isinstance(raw, dict) or "steps" not in raw:
            raise FlowError(f"corrupt flow checkpoint {p}: missing steps")
        flow = cls.__new__(cls)
        flow.flow_id = str(raw.get("flow_id") or ulid_now())
        flow.name = str(raw.get("name") or "")
        flow.steps = [FlowStep.from_dict(s) for s in raw.get("steps") or []]
        flow.checkpoint_path = p
        flow.executor = executor
        flow.index = int(raw.get("index") or 0)
        flow.status = str(raw.get("status") or STATUS_PENDING)
        if flow.status == STATUS_RUNNING:
            # Died mid-run: the in-flight step never recorded a result,
            # so it is safe — and required — to retry it.
            flow.status = STATUS_PAUSED
        flow.results = list(raw.get("results") or [])
        flow.error = str(raw.get("error") or "")
        flow.started_at = float(raw.get("started_at") or 0.0)
        flow.updated_at = float(raw.get("updated_at") or 0.0)
        if not (0 <= flow.index <= len(flow.steps)):
            raise FlowError(
                f"corrupt flow checkpoint {p}: index {flow.index} out of range "
                f"for {len(flow.steps)} steps")
        if len(flow.results) != flow.index and flow.status not in (STATUS_FAILED,):
            # Results and index must agree: one result per completed step.
            # A mismatch means the checkpoint was torn — fail loud.
            raise FlowError(
                f"corrupt flow checkpoint {p}: {len(flow.results)} results "
                f"for index {flow.index}")
        return flow

    # -- execution ------------------------------------------------------------
    def _persist(self) -> None:
        if self.checkpoint_path is not None:
            self.save()

    def run_step(self) -> dict[str, Any]:
        """Run exactly one step (the next unfinished one) and persist.

        Returns the step's result dict. Raises FlowError when the flow is
        finished/failed or has no executor; a step failure marks the flow
        ``failed`` and re-raises the executor's exception with the step
        index attached — the checkpoint still records everything up to
        the failure, so the owner can inspect and resume after fixing
        the cause.
        """
        if self.status == STATUS_DONE:
            raise FlowError("flow is already done — nothing left to run")
        if self.status == STATUS_FAILED:
            raise FlowError(
                f"flow failed at step {self.index}: {self.error} — "
                "fix the cause, reset the error, then resume")
        if self.executor is None:
            raise FlowError("flow has no executor — pass one to run a step")
        step = self.next_step
        if step is None:
            self.status = STATUS_DONE
            self.updated_at = time.time()
            self._persist()
            return {"ok": True, "note": "flow already complete"}
        if self.status == STATUS_PENDING:
            self.status = STATUS_RUNNING
            self.started_at = self.started_at or time.time()
        step_dict = step.to_dict()
        try:
            result = self.executor(step_dict)
        except Exception as exc:
            self.status = STATUS_FAILED
            self.error = f"step {self.index} ({step.kind} {step.target!r}) failed: {exc}"
            self.updated_at = time.time()
            self._persist()
            raise
        if not isinstance(result, dict):
            result = {"ok": True, "value": result}
        self.results.append({"step": self.index, "kind": step.kind,
                             "target": step.target, "result": result,
                             "ts": time.time()})
        self.index += 1
        self.updated_at = time.time()
        if self.index >= len(self.steps):
            self.status = STATUS_DONE
        self._persist()
        return result

    def run_all(self) -> list[dict[str, Any]]:
        """Run steps until done, paused, or failed. Returns the results
        produced by *this* call (not the flow's full history)."""
        out: list[dict[str, Any]] = []
        while self.status in (STATUS_PENDING, STATUS_RUNNING, STATUS_PAUSED):
            if self.status == STATUS_PAUSED:
                break
            if self.next_step is None:
                self.status = STATUS_DONE
                self.updated_at = time.time()
                self._persist()
                break
            out.append(self.run_step())
        return out

    def pause(self) -> None:
        """Pause between steps. A no-op when already paused/pending; a
        finished or failed flow cannot be paused."""
        if self.status in (STATUS_DONE, STATUS_FAILED):
            raise FlowError(f"cannot pause a {self.status} flow")
        if self.status != STATUS_PAUSED:
            self.status = STATUS_PAUSED
            self.updated_at = time.time()
            self._persist()

    def resume(self) -> list[dict[str, Any]]:
        """Continue from the persisted step index. Only a paused flow (or
        one loaded from a checkpoint) can resume."""
        if self.status == STATUS_DONE:
            return []
        if self.status == STATUS_FAILED:
            raise FlowError(
                f"cannot resume a failed flow: {self.error} — "
                "clear the error and reset status to paused first")
        if self.status not in (STATUS_PAUSED, STATUS_PENDING, STATUS_RUNNING):
            raise FlowError(f"cannot resume a flow in status {self.status!r}")
        self.status = STATUS_RUNNING
        self.updated_at = time.time()
        self._persist()
        return self.run_all()

    def reset_failed(self) -> None:
        """Clear a failure so the flow can resume at the failed step
        (the failed step itself is retried — its result was never
        recorded)."""
        if self.status != STATUS_FAILED:
            raise FlowError(f"reset_failed needs a failed flow, got {self.status!r}")
        self.status = STATUS_PAUSED
        self.error = ""
        self.updated_at = time.time()
        self._persist()

    def summary(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "name": self.name,
            "status": self.status,
            "progress": f"{self.index}/{len(self.steps)}",
            "error": self.error,
            "results": len(self.results),
        }
