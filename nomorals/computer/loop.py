"""The desktop control loop: perceive → decide → act → verify.

The browser agent's loop, generalised to the whole desktop. Each
step captures a perception (screenshot + window list + focused
window + pointer), the decider (the brain, or a script) names ONE
action, the action runs through the session's control gate, and
its expected effect is verified against the NEXT perception. An
action whose effect is absent is a reported failure — the loop
stops and says so; it never silently retries, never loops forever:
steps and seconds are budgeted, and every step is journaled with
before/after frame references so the owner can audit exactly what
her computer did.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import failures
from .sessions import GateRefused, SessionManager, SessionRecord

#: Seconds to let the desktop settle between an action and the
#: verification perception.
SETTLE_SECONDS = 0.35


# ------------------------------------------------------------------ types

@dataclass
class Perception:
    """One look at the desktop."""

    session_id: str
    ts: float
    width: int
    height: int
    png: bytes
    frame_path: str
    frame_hash: str
    windows: list[dict[str, Any]]
    focused_id: int
    pointer: dict[str, int]

    def summary(self) -> str:
        lines = [f"screen {self.width}x{self.height}, "
                 f"pointer at {self.pointer.get('x')},{self.pointer.get('y')}, "
                 f"{len(self.windows)} window(s):"]
        for win in self.windows:
            mark = " *focused*" if win.get("focused") else ""
            lines.append(
                f"  - [{win.get('id')}] {win.get('title')!r} "
                f"at {win.get('x')},{win.get('y')} "
                f"{win.get('width')}x{win.get('height')}{mark}")
        return "\n".join(lines)


class Expectation(Protocol):
    """A checkable claim about what an action should change."""

    def describe(self) -> str: ...

    def check(self, before: Perception,
              after: Perception) -> tuple[bool, str]: ...


@dataclass
class Action:
    """One thing to do on the desktop."""

    kind: str  # click|move|drag|scroll|key|type|launch|focus|
    #            close_window|kill|wait
    args: dict[str, Any] = field(default_factory=dict)
    expect: Expectation | None = None
    note: str = ""


@dataclass
class StepRecord:
    n: int
    kind: str
    args: str
    ok: bool
    verified: bool | None
    detail: str
    before_frame: str
    after_frame: str
    ts: float


@dataclass
class RunReport:
    goal: str
    session_id: str
    status: str  # completed|failed|budget_exhausted|gate_refused
    reason: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    started: float = 0.0
    ended: float = 0.0

    def describe(self) -> str:
        lines = [f"computer run — {self.status}"
                 + (f": {self.reason}" if self.reason else "")]
        lines.append(f"goal: {self.goal}")
        for step in self.steps:
            verdict = ("verified" if step.verified else
                       "UNVERIFIED-FAILED" if step.verified is False
                       else "done")
            lines.append(
                f"  {step.n}. {step.kind} {step.args} — {verdict}"
                + (f" ({step.detail})" if step.detail else ""))
        return "\n".join(lines)


# ------------------------------------------------------------- expectations

@dataclass
class TitleIs:
    """A window with exactly this title exists afterwards."""
    title: str

    def describe(self) -> str:
        return f"a window titled {self.title!r} exists"

    def check(self, before: Perception,
              after: Perception) -> tuple[bool, str]:
        titles = [str(w.get("title", "")) for w in after.windows]
        if self.title in titles:
            return True, f"title {self.title!r} present"
        return False, f"titles now: {titles}"


@dataclass
class WindowPresent:
    """A window whose title contains ``substr`` exists afterwards."""
    substr: str

    def describe(self) -> str:
        return f"a window matching {self.substr!r} exists"

    def check(self, before: Perception,
              after: Perception) -> tuple[bool, str]:
        for win in after.windows:
            if self.substr in str(win.get("title", "")):
                return True, f"found {win.get('title')!r}"
        return False, "no matching window"


@dataclass
class WindowGone:
    """No window whose title contains ``substr`` exists afterwards."""
    substr: str

    def describe(self) -> str:
        return f"no window matching {self.substr!r} remains"

    def check(self, before: Perception,
              after: Perception) -> tuple[bool, str]:
        for win in after.windows:
            if self.substr in str(win.get("title", "")):
                return False, f"still open: {win.get('title')!r}"
        return True, "gone"


@dataclass
class PixelsChanged:
    """The frame visibly changed (fraction of pixels, when measurable)."""
    min_fraction: float = 0.0005

    def describe(self) -> str:
        return "the screen visibly changed"

    def check(self, before: Perception,
              after: Perception) -> tuple[bool, str]:
        if before.frame_hash == after.frame_hash:
            return False, "frames are byte-identical"
        fraction = pixel_diff_fraction(before.png, after.png)
        if fraction is None:
            return True, "frames differ (pixel count unavailable)"
        if fraction >= self.min_fraction:
            return True, f"{fraction:.4%} of pixels changed"
        return False, f"only {fraction:.4%} of pixels changed"


@dataclass
class PointerAt:
    x: int
    y: int

    def describe(self) -> str:
        return f"the pointer is at {self.x},{self.y}"

    def check(self, before: Perception,
              after: Perception) -> tuple[bool, str]:
        got = (after.pointer.get("x"), after.pointer.get("y"))
        if got == (self.x, self.y):
            return True, "pointer there"
        return False, f"pointer is at {got[0]},{got[1]}"


def pixel_diff_fraction(png_a: bytes, png_b: bytes) -> float | None:
    """Fraction of differing pixels, or None when Pillow is absent."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return None
    import io
    try:
        a = Image.open(io.BytesIO(png_a)).convert("RGB")
        b = Image.open(io.BytesIO(png_b)).convert("RGB")
    except Exception:  # noqa: BLE001 — unmeasurable is not "changed"
        return None
    if a.size != b.size:
        return 1.0
    diff = ImageChops.difference(a, b).convert("L")
    hist = diff.histogram()
    changed = sum(hist[1:])
    return changed / float(a.size[0] * a.size[1])


# ---------------------------------------------------------------- deciders

class Decider(Protocol):
    """The brain's seat: look at a perception, name the next action.

    Returning None declares the goal complete. Raising aborts the
    run with the error as its reason — loudly, never retried.
    """

    def decide(self, perception: Perception, goal: str,
               history: list[StepRecord]) -> Action | None: ...


class DeciderError(Exception):
    """The decider could not produce a usable action."""


@dataclass
class ScriptedDecider:
    """A fixed action list — programmatic missions and tests."""

    actions: list[Action]

    def decide(self, perception: Perception, goal: str,
               history: list[StepRecord]) -> Action | None:
        if not self.actions:
            return None
        return self.actions.pop(0)


#: Action kinds the loop will execute, mapped to driver methods.
ACTION_METHODS = {
    "click": "mouse_click",
    "move": "mouse_move",
    "drag": "mouse_drag",
    "scroll": "mouse_scroll",
    "key": "key",
    "type": "type",
    "launch": "launch",
    "focus": "focus",
    "close_window": "close_window",
    "kill": "kill",
    # "wait" is loop-local (a pause), never sent to the driver.
}

_LLM_PROMPT = """You are driving a desktop computer to complete a goal.
Reply with ONE JSON object and nothing else:
  {{"action": {{"kind": "click|type|key|launch|focus|move|drag|scroll|close_window|wait", "args": {{...}}}}, "note": "why"}}
or, when the goal is complete:
  {{"done": true, "note": "why"}}
Action args: click/move {{"x","y"}} · type {{"text"}} · key {{"keys": "Return"|"ctrl+l"}}
· launch {{"argv": ["program", "arg"]}} · focus {{"window": <id from the window list>}}
· scroll {{"x","y","dy"}} · wait {{"seconds": 1}}.
Coordinates are absolute screen pixels. Use the window list for
titles and geometry; click centers, not edges.

Goal: {goal}

Current perception:
{perception}

Recent steps:
{history}
"""


@dataclass
class LLMDecider:
    """The brain directs: a completion callable renders each step.

    ``complete_fn`` is injected by the wiring layer (the tool /
    runtime pass the router's completion), so this organ never
    imports the agent layer. The reply must be the JSON action
    object described in the prompt; anything else aborts the run
    as a decider failure — reported, never silently retried.
    """

    complete_fn: Callable[[str], str]

    def decide(self, perception: Perception, goal: str,
               history: list[StepRecord]) -> Action | None:
        history_text = "\n".join(
            f"  {s.n}. {s.kind} {s.args} — "
            f"{'ok' if s.ok else 'FAILED'} {s.detail}"
            for s in history[-5:]) or "  (none yet)"
        prompt = _LLM_PROMPT.format(
            goal=goal, perception=perception.summary(),
            history=history_text)
        try:
            reply = self.complete_fn(prompt)
        except Exception as exc:  # noqa: BLE001 — surfaced as the reason
            raise DeciderError(f"the model call failed: {exc}") from exc
        return parse_action_reply(reply)


def parse_action_reply(reply: str) -> Action | None:
    """Parse an LLMDecider reply into an Action (None = done)."""
    obj = _first_json_object(reply or "")
    if obj is None:
        raise DeciderError(
            f"the decider's reply carried no JSON object: "
            f"{(reply or '')[:200]!r}")
    if obj.get("done"):
        return None
    raw = obj.get("action")
    if not isinstance(raw, dict):
        raise DeciderError(f"decider reply has no action object: {obj!r}")
    kind = str(raw.get("kind") or "")
    if kind not in ACTION_METHODS and kind != "wait":
        raise DeciderError(f"decider named an unknown action kind {kind!r}")
    args = raw.get("args")
    if not isinstance(args, dict):
        args = {}
    return Action(kind=kind, args=args, note=str(obj.get("note") or ""))


def _first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:end + 1])
                    except ValueError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------- the loop

def capture(manager: SessionManager, rec: SessionRecord) -> Perception:
    """One perception: frame + windows + pointer (ungated — eyes)."""
    shot = manager.call(rec, "screenshot")
    png = b""
    if shot.get("data"):
        try:
            png = base64.b64decode(str(shot["data"]))
        except ValueError as exc:
            failures.record_failure("loop capture decode", exc)
    frame_path = ""
    if png:
        frame_path = str(manager.save_frame(rec.id, png))
    wins = manager.call(rec, "windows")
    pointer = manager.call(rec, "pointer")
    return Perception(
        session_id=rec.id, ts=time.time(),
        width=int(shot.get("width") or rec.width),
        height=int(shot.get("height") or rec.height),
        png=png, frame_path=frame_path,
        frame_hash=hashlib.sha256(png).hexdigest() if png else "",
        windows=list(wins.get("windows") or []),
        focused_id=int(wins.get("focused_id") or 0),
        pointer={"x": int(pointer.get("x") or 0),
                 "y": int(pointer.get("y") or 0)})


def action_to_call(action: Action) -> tuple[str, dict[str, Any]]:
    """Map an Action to its driver method + params."""
    method = ACTION_METHODS[action.kind]
    args = dict(action.args)
    if action.kind == "key" and "keys" not in args and "key" in args:
        args["keys"] = args.pop("key")
    return method, args


def run_task(manager: SessionManager, rec: SessionRecord, goal: str,
             decider: Decider, max_steps: int = 25,
             max_seconds: float = 300.0,
             settle_seconds: float = SETTLE_SECONDS) -> RunReport:
    """Drive the desktop toward ``goal`` under the control gate.

    Semantics: one action per step; each verified against the next
    perception when it carries an expectation; the first failure —
    driver error, unmet expectation, decider error, budget, or the
    gate closing mid-run — ends the run with that reason. Nothing
    is retried silently, ever.
    """
    report = RunReport(goal=goal, session_id=rec.id, status="failed",
                       started=time.time())
    manager.journal("run_started", rec.id, goal=goal, max_steps=max_steps)
    try:
        current = manager.reconcile(rec)
        if current.state != "open":
            report.reason = (f"session is {current.state}"
                             + (f" — {current.note}" if current.note
                                else ""))
            return report
        perception = capture(manager, current)
        for n in range(1, max_steps + 1):
            if time.time() - report.started > max_seconds:
                report.status = "budget_exhausted"
                report.reason = (f"time budget of {max_seconds:.0f}s "
                                 "spent")
                return report
            reasons = manager.gate_reasons(current)
            if reasons:
                report.status = "gate_refused"
                report.reason = "; ".join(reasons)
                return report
            try:
                action = decider.decide(perception, goal, report.steps)
            except Exception as exc:  # noqa: BLE001 — the reason, verbatim
                report.reason = f"decider failed: {exc}"
                return report
            if action is None:
                report.status = "completed"
                return report
            step = StepRecord(n=n, kind=action.kind,
                              args=_args_text(action.args), ok=False,
                              verified=None, detail="",
                              before_frame=perception.frame_path,
                              after_frame="", ts=time.time())
            report.steps.append(step)
            try:
                if action.kind == "wait":
                    seconds = float(action.args.get("seconds") or 1)
                    time.sleep(min(max(seconds, 0.0), 30.0))
                else:
                    method, params = action_to_call(action)
                    manager.act(current, method, params)
            except GateRefused as refused:
                step.detail = str(refused)
                report.status = "gate_refused"
                report.reason = str(refused)
                return report
            except Exception as exc:  # noqa: BLE001 — reported, not retried
                step.detail = f"{type(exc).__name__}: {exc}"
                report.reason = (f"action {action.kind} failed: "
                                 f"{step.detail}")
                return report
            step.ok = True
            if settle_seconds:
                time.sleep(settle_seconds)
            after = capture(manager, current)
            step.after_frame = after.frame_path
            if action.expect is not None:
                ok, detail = action.expect.check(perception, after)
                step.verified = ok
                step.detail = detail
                if not ok:
                    report.reason = (
                        f"action {action.kind} did not do what was "
                        f"expected ({action.expect.describe()}): {detail}")
                    return report
            perception = after
        report.status = "budget_exhausted"
        report.reason = f"step budget of {max_steps} spent"
        return report
    finally:
        report.ended = time.time()
        manager.journal("run_finished", rec.id, goal=goal,
                        status=report.status, reason=report.reason,
                        steps=len(report.steps))


def _args_text(args: dict[str, Any]) -> str:
    parts = []
    for key, value in sorted(args.items()):
        parts.append(f"{key}={str(value)[:60]}")
    return " ".join(parts)[:200]
