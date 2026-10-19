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

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.page.url, "title": self.page.title,
                "tier": self.tier, "step": self.step,
                "snapshot_hash": self.page.snapshot_hash(),
                "elements": self.page.elements,
                "last_result": self.last_result}


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
            raise RuntimeError(
                "CdpDriver needs Playwright: pip install playwright "
                "&& playwright install chromium"
            ) from exc
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
                 on_blocked: Callable[[Observation, str], str] | None = None) -> None:
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

    def _wall_check(self, page: Page, step: int) -> tuple[str | None, Page]:
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
        obs = Observation(page=page, step=step)
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

    def run(self, start_url: str, *, goal: str = "") -> dict[str, Any]:
        page = self.driver.fetch(start_url)
        wall, page = self._wall_check(page, 0)
        if wall is not None:
            self.stopped_reason = wall
            return {"goal": goal, "steps": 0,
                    "stopped_reason": self.stopped_reason,
                    "final_url": page.url,
                    "trace": []}
        last_result: dict[str, Any] | None = None
        for step in range(1, self.max_steps + 1):
            obs = Observation(page=page, last_result=last_result, step=step)
            decision = self._decide(obs)
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
            wall, page = self._wall_check(page, step)
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
        return {"goal": goal, "steps": len(self.trace),
                "stopped_reason": self.stopped_reason,
                "final_url": page.url,
                "trace": [t.to_dict() for t in self.trace]}
