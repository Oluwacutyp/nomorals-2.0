"""Production wiring: ``AgentLoop`` + ``ChallengeSolver`` + takeover.

Until this module, the loop's ``on_blocked`` seam had NO production
caller — the takeover library, the Telegram notifier, and the
``/browser takeover`` verb all existed, but when Devon herself hit a
wall mid-task she never auto-escalated. ``build_wired_agent_loop``
closes that seam: one live tab, one solver with every seam attached
(takeover manager, owner notifier, challenge memory, solved-session
store), and the loop's ``on_blocked`` set to the solver's handler.

Flow when the loop hits a wall:

1. the solver works the ladder (retry -> wait -> captcha service ->
   fresh session), self-solve first;
2. only when the ladder is exhausted does it escalate — the Telegram
   escalation (attempt receipt + one-time link + webview/Chrome hint)
   goes to the OWNER chat via the injected notifier;
3. the loop PAUSES while the owner solves (``wait_for_resolution``);
   on a verified clear it resumes where it left off, on the
   post-clear page — never on the stale wall HTML;
4. still-blocked stops the loop honestly and saves nothing.

The driver drives the SAME live tab the solver sees, so a resume
continues the same page the owner just cleared — not a copy.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .agent import (
    Action,
    AgentLoop,
    BrowserDriver,
    Page,
)
from .challenge import ChallengeSolver
from .service import RenderedTabTakeoverAdapter
from .vision_marks import PerceptionEscalator

_log = get_logger(__name__)

__all__ = [
    "RenderedTabDriver",
    "build_vision_escalator",
    "build_wired_agent_loop",
]

_REF_LINE = re.compile(r"\[ref=([^\]]+)\]")


class RenderedTabDriver:
    """A :class:`BrowserDriver` over one live rendered tab.

    Duck-typed: anything with ``navigate`` / ``html`` / ``click`` /
    ``fill`` / ``snapshot`` works — the real ``RenderedTab`` or a test
    fake. Missing capabilities raise an honest error instead of
    pretending the action worked; the loop records the failure.
    """

    def __init__(self, tab: Any) -> None:
        self.tab = tab

    # -- BrowserDriver ----------------------------------------------------
    def fetch(self, url: str) -> Page:
        navigate = getattr(self.tab, "navigate", None)
        if callable(navigate) and url:
            navigate(url)
        return self._read()

    def act(self, action: Action, page: Page) -> Page:
        kind = (action.kind or "").lower()
        target = action.target or ""
        if kind == "goto":
            return self.fetch(target or action.payload.get("url", ""))
        if kind == "click":
            self._call("click", target)
            return self._read()
        if kind == "fill":
            value = action.payload.get("value", "")
            if not self._try("fill", target, value):
                self._call("type_text", target, value)
            return self._read()
        if kind == "submit":
            if not self._try("submit", target):
                self._call("click", target)
            return self._read()
        if kind == "select":
            self._call("select", target, action.payload.get("value", ""))
            return self._read()
        if kind == "check":
            self._call("check", target, True)
            return self._read()
        if kind == "wait":
            return self._read()
        if kind == "back":
            self._call("back")
            return self._read()
        raise RuntimeError(f"driver cannot perform action {kind!r}")

    # -- page reading -------------------------------------------------------
    def _read(self) -> Page:
        raw = self.tab.html()
        if isinstance(raw, dict):
            html = str(raw.get("html") or "")
            url = str(raw.get("url") or getattr(self.tab, "url", "") or "")
            title = str(raw.get("title") or getattr(self.tab, "title", "")
                        or "")
        else:
            html = str(raw or "")
            url = str(getattr(self.tab, "url", "") or "")
            title = str(getattr(self.tab, "title", "") or "")
        return Page(url=url, html=html, title=title,
                    elements=self._elements())

    def _elements(self) -> list[dict[str, Any]]:
        """Snapshot refs -> loop elements. Best effort: [] on failure."""
        snapshot = getattr(self.tab, "snapshot", None)
        if not callable(snapshot):
            return []
        try:
            snap = snapshot()
        except Exception:  # noqa: BLE001 - snapshot failure is not fatal
            return []
        text = snap.get("snapshot", "") if isinstance(snap, dict) else ""
        elements: list[dict[str, Any]] = []
        for line in str(text).splitlines():
            match = _REF_LINE.search(line)
            if not match:
                continue
            label = _REF_LINE.sub("", line).strip(" -•\t")
            elements.append({"id": f"ref:{match.group(1)}",
                             "label": label})
        return elements

    # -- tab calls ------------------------------------------------------------
    def _try(self, name: str, *args: Any) -> bool:
        fn = getattr(self.tab, name, None)
        if not callable(fn):
            return False
        fn(*args)
        return True

    def _call(self, name: str, *args: Any) -> None:
        fn = getattr(self.tab, name, None)
        if not callable(fn):
            raise RuntimeError(
                f"rendered tab has no {name}() for this action")
        fn(*args)


def build_wired_agent_loop(*, tab: Any,
                           takeover_manager: Any,
                           notifier: Callable[[str, str], Any] | None,
                           memory: Any = None,
                           session_store: Any = None,
                           owner_chat: str = "",
                           task_label: str = "",
                           resume_state: dict[str, Any] | None = None,
                           max_steps: int = 30,
                           autonomy: str = "propose",
                           decide_llm: Any = None,
                           approve: Any = None,
                           verify: Any = None,
                           dry_run: bool = False,
                           on_trace: Any = None,
                           on_progress: Any = None,
                           on_wall_outcome: Any = None,
                           cancel_requested: Any = None,
                           vision_brain: Any = None,
                           escalator: Any = None) -> AgentLoop:
    """Build an ``AgentLoop`` whose wall path is fully wired.

    ``tab`` is the live rendered tab (raw — the factory wraps it in
    the takeover adapter for the solver and drives it with
    :class:`RenderedTabDriver` for the loop). ``notifier`` is the
    owner-chat sender (``(message, url) -> None``); ``session_store``
    is where cleared sessions' cookies persist. ``on_wall_outcome``
    (optional) receives wall lifecycle events — ``wall_hit`` /
    ``wall_escalated`` / ``wall_cleared`` / ``wall_blocked`` — so a
    background job can notify the owner and record the wall outcome.
    ``cancel_requested`` (optional) is the cooperative cancel poll
    handed to the loop — the browser job runner passes its job's
    ``cancel_requested`` so a cancel stops the run at its next step
    boundary. ``vision_brain`` (optional) is the brain whose vision
    rung (``describe_image``) becomes the perception escalator's
    ``see``: the loop then escalates to Set-of-Mark vision when the
    text snapshot cannot find the target, instead of deciding
    blind. ``escalator`` (optional) injects a prebuilt escalator
    instead (tests, custom extractors). Everything else is passed
    straight through to :class:`AgentLoop`.
    """
    adapter = RenderedTabTakeoverAdapter(tab)
    solver = ChallengeSolver(
        memory=memory, takeover_manager=takeover_manager,
        notifier=notifier, session_store=session_store)
    handler = solver.wall_handler_for(
        tab=adapter, resume_state=dict(resume_state or {}),
        owner_chat=owner_chat, task_label=task_label or "browsing",
        on_outcome=on_wall_outcome)
    if escalator is None and vision_brain is not None:
        escalator = build_vision_escalator(tab, vision_brain)
    loop = AgentLoop(
        RenderedTabDriver(tab), decide_llm=decide_llm,
        autonomy=autonomy, approve=approve, verify=verify,
        dry_run=dry_run, max_steps=max_steps,
        on_trace=on_trace, on_progress=on_progress,
        on_blocked=handler, cancel_requested=cancel_requested,
        escalator=escalator)
    return loop


def build_vision_escalator(tab: Any, brain: Any) -> PerceptionEscalator:
    """A :class:`PerceptionEscalator` with live eyes for one tab.

    ``see`` is the brain's vision rung: the marked screenshot's bytes
    go to ``describe_image`` and its text comes back. ``screenshot``
    is the tab's own capture (a file path, per the rendered tab's
    ``screenshot()`` shape). Both are lazy: nothing is captured or
    asked until the loop actually escalates.
    """

    def _screenshot() -> str:
        fn = getattr(tab, "screenshot", None)
        if not callable(fn):
            raise RuntimeError("this tab cannot take a screenshot")
        out = fn()
        path = out.get("path") if isinstance(out, dict) else ""
        if not path:
            raise RuntimeError("the tab's screenshot produced no file")
        return str(path)

    def _see(path: str, prompt: str) -> str:
        data = Path(path).read_bytes()
        resp = brain.describe_image(data, prompt)
        if getattr(resp, "ok", True) is False:
            raise RuntimeError(
                f"vision model declined: "
                f"{getattr(resp, 'error', '') or 'no answer'}")
        text = str(getattr(resp, "text", "") or "")
        if not text:
            raise RuntimeError("vision model returned an empty answer")
        return text

    return PerceptionEscalator(see=_see, screenshot=_screenshot)
