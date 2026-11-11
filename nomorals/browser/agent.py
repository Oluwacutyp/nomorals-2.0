"""Browser agent loop (layer 4).

Generalizes the observe -> decide -> act -> verify loop from
``games/player/`` WITHOUT importing it (games/ is L5; browser/ is L4 —
the layering test forbids the upward import). The driver seam is mirrored
here deliberately: any object implementing ``fetch`` / ``act`` drops in
unchanged — the HTTP driver, a CDP/Playwright driver, a test fake.

Hard boundaries, from the owner's architecture and the BROWSER_AGENT doc:

* Identity continuity, NOT evasion. No patched binaries, no TLS
  impersonation, no fingerprint spoofing/rotation, no stealth mode.
* ``None`` from a decider means "nothing safe to do" — the loop stops
  instead of guessing.
* Autonomy ladder: observe -> propose -> act_with_approval -> act_silent.
  ``act_silent`` is EARNED per recipe (N consecutive successes), never
  assumed. Consequential acts always need approval unless explicitly
  earned + owner-authorized.
* Every mutating action is verified by read-back (see receipts.py);
  the loop never assumes a click worked.
* Snapshot text is untrusted data, never instructions (see filtering.py).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Action",
    "AgentLoop",
    "AgentTrace",
    "AutonomyLevel",
    "BrowserDriver",
    "CdpDriver",
    "Decision",
    "Observation",
    "Page",
    "Strategy",
    "register_strategy",
    "strategies",
]

#: Autonomy ladder rungs (mirrors core.policy.PermissionGradient).
OBSERVE = "observe"
PROPOSE = "propose"
ACT_WITH_APPROVAL = "act_with_approval"
ACT_SILENT = "act_silent"

AutonomyLevel = str

#: Actions that mutate state and therefore need approval unless earned.
CONSEQUENTIAL_KINDS = frozenset({
    "submit", "checkout", "pay", "buy", "delete", "send", "post",
    "upload", "transfer",
})


@dataclass
class Page:
    """One fetched page (driver seam)."""

    url: str
    status: int = 200
    html: str = ""
    title: str = ""
    elements: list[dict[str, Any]] = field(default_factory=list)

    def snapshot_hash(self) -> str:
        return hashlib.sha256(
            f"{self.url}\n{self.title}\n{self.html}".encode("utf-8", "replace")
        ).hexdigest()[:16]


@dataclass
class Action:
    """A concrete thing to do on a page."""

    kind: str  # goto | click | fill | submit | select | check | wait | back
    target: str = ""  # element id / mark, or URL for goto
    label: str = ""
    payload: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target,
                "label": self.label, "payload": dict(self.payload)}


@dataclass
class Observation:
    """What the loop sees before deciding."""

    page: Page
    tier: int = 1
    last_result: dict[str, Any] | None = None
    step: int = 0
    #: The run's goal, so a decider (e.g. the LLM decider) can judge
    #: whether a candidate action advances it instead of guessing.
    goal: str = ""
    #: Steps taken so far: [{step, action, result}]. The decider sees
    #: what already happened — no repeating a failed move blind.
    history: list[dict[str, Any]] = field(default_factory=list)
    #: Vision escalation (Tier 3, vision_marks.py): what the vision
    #: model reported from a marked screenshot of this page, and the
    #: SoM overlay it answered over. Empty/None on the plain text
    #: path — the loop only fills these when perception escalates,
    #: so a run that never escalates is byte-identical to before.
    vision_answer: str = ""
    som: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"url": self.page.url, "title": self.page.title,
               "tier": self.tier, "step": self.step,
               "goal": self.goal, "history": list(self.history),
               "snapshot_hash": self.page.snapshot_hash(),
               "elements": self.page.elements,
               "last_result": self.last_result}
        if self.vision_answer:
            out["vision_answer"] = self.vision_answer
        if self.som is not None:
            out["som"] = self.som
        return out


@dataclass
class Decision:
    """A decided action, with why, and whether it is hard to undo."""

    action: Action
    reason: str
    irreversible: bool = False
    confidence: float = 0.5

    def needs_approval(self, autonomy: AutonomyLevel) -> bool:
        if autonomy == ACT_SILENT:
            return False
        if self.irreversible:
            return True
        if self.action.kind in CONSEQUENTIAL_KINDS:
            return True
        return autonomy in (OBSERVE, PROPOSE)


class BrowserDriver(Protocol):
    """Two-method seam. Any driver implementing this drops in unchanged."""

    def fetch(self, url: str) -> Page: ...
    def act(self, action: Action, page: Page) -> Page: ...


class CdpDriver:
    """CDP/Playwright driver over the real browser.

    Playwright is imported lazily — importing this module never requires
    it. When Playwright is absent, construction raises an honest error
    naming what is missing instead of pretending to drive a browser.
    """

    def __init__(self, *, headless: bool = True, **_kwargs: Any) -> None:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            # Self-provision (owner doctrine): a missing engine is
            # Devon's job to fix — one bounded install attempt, then
            # the import is retried before the honest error stands.
            from .provision import ensure_playwright

            if not ensure_playwright():
                raise RuntimeError(
                    "CdpDriver needs Playwright: the package and its "
                    "chromium build are missing, and self-provisioning "
                    "could not install them on this host/profile"
                ) from exc
            from playwright.sync_api import sync_playwright  # type: ignore
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._context = self._browser.new_context()
        self._page = self._context.new_page()
        self._headless = headless

    def fetch(self, url: str) -> Page:
        self._page.goto(url)
        return Page(url=self._page.url, status=200,
                    html=self._page.content(), title=self._page.title())

    def act(self, action: Action, page: Page) -> Page:
        if action.kind == "goto":
            return self.fetch(action.target or action.payload.get("url", ""))
        if action.kind == "click" and action.target:
            self._page.click(action.target)
        elif action.kind == "fill" and action.target:
            self._page.fill(action.target,
                            action.payload.get("value", ""))
        elif action.kind == "back":
            self._page.go_back()
        return Page(url=self._page.url, status=200,
                    html=self._page.content(), title=self._page.title())

    def close(self) -> None:
        try:
            self._browser.close()
            self._playwright.stop()
        except Exception:  # noqa: BLE001 - teardown best effort
            pass


Strategy = Callable[[Observation], Decision | None]
strategies: dict[str, Strategy] = {}


def register_strategy(name: str, fn: Strategy) -> None:
    strategies[name.lower()] = fn


def _heuristic_strategy(obs: Observation) -> Decision | None:
    """Conservative fallback: never spends, never deletes, never posts.

    Prefers free gains (claim/collect/daily) and otherwise returns None —
    "nothing safe to do" — instead of guessing.
    """
    avoid = ("delete", "destroy", "pay", "checkout", "subscribe",
             "withdraw", "deposit", "remove account", "top up")
    gain = ("claim", "collect", "daily", "check in", "check-in", "earn", "free")
    for el in obs.page.elements:
        label = str(el.get("label", "")).lower()
        if any(k in label for k in avoid):
            continue
        if any(k in label for k in gain):
            return Decision(Action(kind="click", target=str(el.get("id", "")),
                                   label=str(el.get("label", ""))),
                            reason="heuristic: free gain", confidence=0.6)
    return None


@dataclass
class AgentTrace:
    """One loop step: before-hash, action, after-hash, result, ms."""

    step: int
    before_url: str
    before_hash: str
    action: dict[str, Any]
    after_url: str
    after_hash: str
    result: dict[str, Any]
    ms: float
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "before_url": self.before_url,
                "before_hash": self.before_hash, "action": self.action,
                "after_url": self.after_url, "after_hash": self.after_hash,
                "result": self.result, "ms": self.ms,
                "verified": self.verified}


class AgentLoop:
    """observe -> decide -> act -> verify -> record.

    ``decide_llm`` is an injected callable (Observation -> Decision|None).
    When it returns None, the heuristic fallback runs; when that returns
    None, the loop stops honestly. No LLM is imported here — the caller
    supplies the brain.
    """

    def __init__(self, driver: BrowserDriver, *,
                 decide_llm: Callable[[Observation], Decision | None] | None = None,
                 strategy: str = "",
                 autonomy: AutonomyLevel = PROPOSE,
                 approve: Callable[[Decision, Observation], bool] | None = None,
                 verify: Callable[[Action, Page, Page], bool] | None = None,
                 dry_run: bool = False,
                 max_steps: int = 30,
                 on_trace: Callable[[AgentTrace], None] | None = None,
                 on_progress: Callable[[dict[str, Any]], None] | None = None,
                 on_blocked: Callable[[Observation, str], str] | None = None,
                 cancel_requested: Callable[[], bool] | None = None,
                 escalator: Any = None) -> None:
        self.driver = driver
        self.decide_llm = decide_llm
        self.strategy = strategy.lower()
        self.autonomy = autonomy
        self.approve = approve
        self.verify = verify or (lambda _a, _b, _c: True)
        self.dry_run = dry_run
        self.max_steps = max(1, int(max_steps))
        self.on_trace = on_trace
        self.on_progress = on_progress
        #: Wall handler: called with (observation, detection) when a
        #: challenge page appears. Return "continue" to keep going
        #: (the handler cleared it, e.g. via ChallengeSolver +
        #: takeover); anything else stops the loop honestly.
        #: No handler + wall = stop, never blindly continue.
        self.on_blocked = on_blocked
        #: Cooperative cancel hook (doc 18 P1): a zero-arg callable
        #: the loop polls at step boundaries — wired by the browser
        #: job runner to ``BrowserRunJob.cancel_requested``. None
        #: means the loop runs unobserved, exactly as before.
        self.cancel_requested = cancel_requested
        #: Perception escalator (vision_marks.PerceptionEscalator),
        #: injected by the wiring layer with a live ``see`` (the
        #: brain's vision rung) and a screenshot source. None = the
        #: text-only path, exactly as before this seam existed.
        self.escalator = escalator
        #: How many steps this run escalated to vision. Journaled via
        #: on_progress as it happens; the count survives on the loop.
        self.vision_escalations = 0
        self.trace: list[AgentTrace] = []
        self.stopped_reason = ""

    def _decide(self, obs: Observation) -> Decision | None:
        if self.strategy and self.strategy in strategies:
            got = strategies[self.strategy](obs)
            if got is not None:
                return got
        if self.decide_llm is not None:
            got = self.decide_llm(obs)
            if got is not None:
                return got
        return _heuristic_strategy(obs)

    def _escalate_if_needed(
            self, obs: Observation,
            decision: Decision | None) -> tuple[Decision | None, Observation]:
        """Vision escalation inside the observe step.

        When the escalator's own rules fire (target not found, the
        same click failed twice, a canvas/WebGL page, a thin
        snapshot), climb the tiers: Tier 2 re-decides over the
        deeper structural element list when an extractor is wired;
        Tier 3 captures the marked screenshot, asks the injected
        vision rung what it sees, and either turns its mark answer
        into this step's decision or hands the vision verdict to
        the decider (``obs.vision_answer`` / ``obs.som``) for a
        second decide. Escalation use is journaled (log +
        on_progress). Any escalation failure is recorded in the
        failure ledger and the plain decision stands — vision is a
        rung, never a gate, and a run with no escalator attached
        never enters this path at all.
        """
        esc = self.escalator
        if esc is None:
            return decision, obs
        page = obs.page
        html = page.html or ""
        last = self.trace[-1] if self.trace else None
        last_click_failed = bool(
            last is not None
            and last.action.get("kind") == "click"
            and not last.result.get("ok", True))
        click_target = str(last.action.get("target", "")) \
            if last is not None else ""
        try:
            needed = esc.should_escalate(
                target_found=decision is not None,
                last_click_failed=last_click_failed,
                is_canvas="<canvas" in html.lower(),
                snapshot_thin=not page.elements and len(html) >= 1000,
                click_target=click_target if last_click_failed else "")
        except Exception:  # noqa: BLE001 - the rule never gates deciding
            _log.debug("vision escalation rule failed", exc_info=True)
            return decision, obs
        if not needed:
            return decision, obs
        try:
            from .vision_marks import TIER_SOM_VISION, TIER_STRUCTURAL

            if esc.tier < TIER_STRUCTURAL:
                esc.escalate()
            if esc.tier == TIER_STRUCTURAL:
                richer = esc.structural_elements()
                if richer:
                    elements = [
                        {**el, "id": str(el.get("id")
                                         or el.get("selector") or "")}
                        for el in richer]
                    obs = Observation(
                        page=Page(url=page.url, status=page.status,
                                  html=page.html, title=page.title,
                                  elements=elements),
                        tier=TIER_STRUCTURAL,
                        last_result=obs.last_result, step=obs.step,
                        goal=obs.goal, history=obs.history)
                    decision = self._decide(obs)
                    if decision is not None:
                        self._note_vision_use(obs, TIER_STRUCTURAL, "")
                        return decision, obs
                esc.escalate()
            if esc.tier >= TIER_SOM_VISION:
                overlay = esc.som_view(list(obs.page.elements),
                                       goal=obs.goal)
                answer = esc.see_marks(overlay)
                obs.vision_answer = answer
                obs.som = overlay.to_dict()
                obs.tier = TIER_SOM_VISION
                self._note_vision_use(obs, TIER_SOM_VISION, answer)
                marked = esc.parse_mark(answer, overlay.marks)
                if marked is not None and marked.selector:
                    return Decision(
                        Action(kind="click", target=marked.selector,
                               label=marked.label),
                        reason=(f"vision escalation: the model picked "
                                f"mark {marked.mark} "
                                f"({marked.label or 'unlabeled'}) from "
                                "the marked screenshot"),
                        confidence=0.6), obs
                # No mark named: the vision verdict still reaches the
                # decider, which re-decides having "seen" the page.
                return self._decide(obs), obs
        except Exception as exc:  # noqa: BLE001 - journaled, never fatal
            try:
                from .failures import record_failure

                record_failure("browser.agent.vision_escalation", exc)
            except Exception:  # noqa: BLE001 - ledger never gates
                pass
            _log.warning("vision escalation failed; continuing on "
                         "the text path", exc_info=True)
        return decision, obs

    def _note_vision_use(self, obs: Observation, tier: int,
                         answer: str) -> None:
        """Journal one escalation: count + log + progress event."""
        self.vision_escalations += 1
        _log.info("browser loop vision escalation at step %s (tier %s)",
                  obs.step, tier)
        if self.on_progress is not None:
            try:
                self.on_progress({
                    "step": obs.step, "url": obs.page.url,
                    "vision_escalation": {
                        "tier": tier, "answer": (answer or "")[:200]}})
            except Exception:  # noqa: BLE001 - progress never gates
                _log.debug("vision escalation progress note failed",
                           exc_info=True)

    def _wall_check(self, page: Page, step: int, goal: str = "") -> tuple[str | None, Page]:
        """Detect a wall on ``page``; run the handler; return stop reason.

        Returns ``(None, current_page)`` when there is no wall, or the
        handler cleared it ("continue"). Any other outcome stops the
        loop honestly with ``(reason, page)``.

        On a cleared wall the returned page is FRESHLY RE-FETCHED from
        the driver: the handler (solver / takeover) cleared the wall on
        the LIVE tab, so the loop must resume on the post-clear page —
        deciding on the stale wall HTML would re-decide a page that no
        longer exists. A refetch failure keeps the old page (best
        effort, logged) rather than stopping a cleared run.
        """
        try:
            from .errors import detect_challenge as _detect
            detection = _detect(page.html or "", title=page.title or "")
        except Exception:  # noqa: BLE001 - detection failure is not a wall
            return None, page
        if not detection:
            return None, page
        if self.on_blocked is None:
            return f"blocked: {detection} (no wall handler)", page
        obs = Observation(page=page, step=step, goal=goal,
                          history=[t.to_dict() for t in self.trace])
        try:
            verdict = self.on_blocked(obs, detection)
        except Exception as exc:  # noqa: BLE001 - handler errors stop us
            return (f"blocked: {detection} (wall handler failed: {exc})",
                    page)
        if verdict == "continue":
            try:
                return None, self.driver.fetch(page.url)
            except Exception:  # noqa: BLE001 - refetch is best effort
                _log.debug("wall cleared but refetch failed; keeping "
                           "stale page", exc_info=True)
                return None, page
        return f"blocked: {detection} (wall handler said stop)", page

    def _cancel_seen(self) -> bool:
        """Poll the injected cancel hook. Fail-open: a hook that
        raises is recorded once in the browser failure ledger, the
        poll is disarmed for the rest of the run, and the run
        continues exactly as an unobserved loop behaves — a broken
        cancel read must never kill a healthy run."""
        hook = self.cancel_requested
        if hook is None:
            return False
        try:
            return bool(hook())
        except Exception as exc:  # noqa: BLE001 - fail-open poll
            self.cancel_requested = None
            try:
                from .failures import record_failure

                record_failure("browser.agent.cancel_poll", exc)
            except Exception:  # noqa: BLE001 - ledger never gates
                pass
            _log.warning("browser agent cancel poll failed; "
                         "continuing unobserved", exc_info=True)
            return False

    def run(self, start_url: str, *, goal: str = "") -> dict[str, Any]:
        armed = self.cancel_requested is not None
        if self._cancel_seen():
            # Cancelled before the first page operation: nothing
            # has been driven yet, so nothing is in flight.
            self.stopped_reason = "cancelled: owner request"
            out = {"goal": goal, "steps": 0,
                   "stopped_reason": self.stopped_reason,
                   "final_url": start_url, "trace": []}
            if armed:
                out["cancel_poll_armed"] = True
            return out
        page = self.driver.fetch(start_url)
        wall, page = self._wall_check(page, 0, goal)
        if wall is not None:
            self.stopped_reason = wall
            out = {"goal": goal, "steps": 0,
                   "stopped_reason": self.stopped_reason,
                   "final_url": page.url,
                   "trace": []}
            if armed:
                out["cancel_poll_armed"] = True
            return out
        last_result: dict[str, Any] | None = None
        cancelled = False
        for step in range(1, self.max_steps + 1):
            if self._cancel_seen():
                # Step boundary: the previous action completed and
                # was verified; no page operation is in flight, so
                # stopping here corrupts nothing.
                cancelled = True
                self.stopped_reason = "cancelled: owner request"
                break
            obs = Observation(page=page, last_result=last_result, step=step,
                              goal=goal,
                              history=[t.to_dict() for t in self.trace])
            decision = self._decide(obs)
            decision, obs = self._escalate_if_needed(obs, decision)
            if decision is None:
                self.stopped_reason = "nothing safe to do"
                break
            if decision.needs_approval(self.autonomy):
                if self.approve is None or not self.approve(decision, obs):
                    self.stopped_reason = "approval denied / no approver"
                    break
            before_hash = page.snapshot_hash()
            before_url = page.url
            started = time.monotonic()
            if self.dry_run:
                new_page = page
                ok = True
            else:
                try:
                    new_page = self.driver.act(decision.action, page)
                    ok = True
                except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                    new_page = page
                    ok = False
                    last_result = {"ok": False, "error": str(exc)}
            ms = (time.monotonic() - started) * 1000.0
            verified = bool(ok and self.verify(decision.action, page, new_page)) \
                if not self.dry_run else True
            entry = AgentTrace(step=step, before_url=before_url,
                               before_hash=before_hash,
                               action=decision.action.to_dict(),
                               after_url=new_page.url,
                               after_hash=new_page.snapshot_hash(),
                               result={"ok": ok, "verified": verified},
                               ms=ms, verified=verified)
            self.trace.append(entry)
            if self.on_trace is not None:
                self.on_trace(entry)
            if self.on_progress is not None:
                self.on_progress({"step": step, "goal": goal,
                                  "url": new_page.url, "verified": verified})
            page = new_page
            last_result = {"ok": ok, "verified": verified}
            wall, page = self._wall_check(page, step, goal)
            if wall is not None:
                self.stopped_reason = wall
                break
            if self.dry_run:
                # Dry run previews ONE move without acting. Continuing
                # would re-decide the same unchanged page until the
                # budget burned — a dry run that loops is a lie.
                self.stopped_reason = "dry_run: preview only, no action taken"
                break
            if not ok:
                self.stopped_reason = f"action failed: {last_result.get('error', '')}"
                break
        else:
            self.stopped_reason = "step budget exhausted"
        if not cancelled \
                and not self.stopped_reason.startswith("blocked:") \
                and self._cancel_seen():
            # Close-out re-check: the request landed after the
            # final step boundary, while the loop was closing —
            # the run reads cancelled, not completed. A blocked
            # stop is exempt: the wall is the run's own terminal
            # truth, and the record says so.
            prior = self.stopped_reason
            self.stopped_reason = "cancelled: owner request" + (
                f" (loop had stopped: {prior})" if prior else "")
        out = {"goal": goal, "steps": len(self.trace),
               "stopped_reason": self.stopped_reason,
               "final_url": page.url,
               "trace": [t.to_dict() for t in self.trace]}
        if armed:
            # Tells the job's finish() this runner polled
            # cooperatively, so a request standing at record
            # close still lands the run cancelled (jobs.py).
            out["cancel_poll_armed"] = True
        return out
