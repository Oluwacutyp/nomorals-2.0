"""Long tasks + parallel browsing + owner takeover.

* Durable queue + checkpoint/resume: a long task is a queue job with a
  checkpoint after every step, so it resumes instead of restarting.
  The queue is injected (storage/queue.py WorkQueue shape) — this module
  never imports storage directly in the hot path, so tests use a fake.
* Streaming progress to chat via an injected ``on_progress`` callback.
* Parallel tabs for independent READS (bounded by injected concurrency
  guard); MUTATING steps are serialized through an injected lock /
  congestion guard (planning/congestion.py shape) — never parallel.
* Live view + owner takeover mid-task: the owner fixes one stuck step
  (see handoff.py's HandoffSession), then hands back and the agent
  resumes from exactly that step.
* Designed handoffs for still-fail cases: canvas/WebGL, passkeys +
  hardware, device-bound banking, full bot protection, escalating
  CAPTCHAs, anything needing a body. A handoff is screenshot + exact
  step + session held warm — not a failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .handoff import HandoffSession

_log = get_logger(__name__)

__all__ = ["DesignedHandoff", "LongTask", "LongTaskRunner", "STILL_FAIL_KINDS",
           "parallel_reads"]

#: Cases that are handed off by design, never ground through.
STILL_FAIL_KINDS = frozenset({
    "canvas_webgl", "passkey_hardware", "device_bound_banking",
    "full_bot_protection", "escalating_captcha", "needs_body",
})


@dataclass
class DesignedHandoff:
    """A handoff that is a designed outcome, not a failure."""

    kind: str
    step: int
    exact_step: str
    screenshot: str = ""
    session_warm: bool = True
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "step": self.step,
                "exact_step": self.exact_step, "screenshot": self.screenshot,
                "session_warm": self.session_warm, "detail": self.detail,
                "handoff": True}


@dataclass
class LongTask:
    """A long task: ordered steps, checkpoint after each."""

    name: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "steps": list(self.steps),
                "step": self.step, "state": dict(self.state)}


def parallel_reads(urls: list[str], *,
                   fetch: Callable[[str], dict[str, Any]],
                   guard: Callable[[str], Any] | None = None) -> list[dict[str, Any]]:
    """Independent READS in parallel (bounded by the caller's guard).

    This helper is sequential in-process; the caller supplies a guard
    (DomainConcurrency slot / thread pool) for real parallelism. Reads
    only — mutating steps must never come through here.
    """
    results: list[dict[str, Any]] = []
    for url in urls:
        if guard is not None:
            with guard(url):
                results.append(fetch(url))
        else:
            results.append(fetch(url))
    return results


class LongTaskRunner:
    """Run a LongTask with checkpoints, progress, and owner takeover."""

    def __init__(self, *,
                 execute_step: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
                 mutate_lock: Callable[[], Any] | None = None,
                 checkpoint: Callable[[LongTask], None] | None = None,
                 on_progress: Callable[[dict[str, Any]], None] | None = None,
                 handoff_session: HandoffSession | None = None) -> None:
        self._execute_step = execute_step
        self._mutate_lock = mutate_lock
        self._checkpoint = checkpoint
        self._on_progress = on_progress
        self.handoff = handoff_session or HandoffSession()

    def run(self, task: LongTask) -> dict[str, Any]:
        """Run from task.step (resume-friendly). Returns outcome dict."""
        while task.step < len(task.steps):
            step = task.steps[task.step]
            kind = str(step.get("kind", ""))
            if kind in STILL_FAIL_KINDS:
                return {"ok": False, "handoff": DesignedHandoff(
                    kind=kind, step=task.step,
                    exact_step=str(step.get("description", step)),
                    screenshot=str(step.get("screenshot", "")),
                    detail=f"designed handoff for {kind}").to_dict(),
                    "steps_run": task.step}
            is_mutating = bool(step.get("mutating", False))
            if is_mutating and self._mutate_lock is not None:
                with self._mutate_lock():
                    result = self._execute_step(step, task.state)
            else:
                result = self._execute_step(step, task.state)
            task.state.update(result.get("state", {}) or {})
            task.step += 1
            if self._checkpoint is not None:
                self._checkpoint(task)
            if self._on_progress is not None:
                self._on_progress({"task": task.name, "step": task.step,
                                   "of": len(task.steps),
                                   "ok": bool(result.get("ok", True))})
            if not result.get("ok", True):
                if result.get("needs_owner"):
                    return {"ok": False, "handoff": DesignedHandoff(
                        kind="needs_body", step=task.step - 1,
                        exact_step=str(step.get("description", step)),
                        detail=str(result.get("error", ""))).to_dict(),
                        "steps_run": task.step}
                return {"ok": False, "error": result.get("error", "step failed"),
                        "failed_step": task.step - 1, "steps_run": task.step}
        return {"ok": True, "steps_run": task.step, "state": dict(task.state)}

    def takeover(self, *, note: str = "") -> dict[str, Any]:
        """Owner takes the mouse mid-task; agent resumes after hand-back."""
        try:
            self.handoff.start_live()
            self.handoff.pause()
            self.handoff.hand_mouse()
        except Exception as exc:  # noqa: BLE001 - handoff state is data
            return {"ok": False, "error": f"takeover failed: {exc}"}
        return {"ok": True, "state": "handed_to_owner", "note": note}

    def hand_back(self) -> dict[str, Any]:
        """Owner hands back; agent resumes from the checkpointed step."""
        try:
            self.handoff.resume()
            self.handoff.finish()
        except Exception as exc:  # noqa: BLE001 - handoff state is data
            return {"ok": False, "error": f"hand-back failed: {exc}"}
        return {"ok": True, "state": "resumed"}
