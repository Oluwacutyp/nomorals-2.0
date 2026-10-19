"""Challenge detection + self-solving, takeover LAST.

When Devon hits a wall (Cloudflare / Turnstile / CAPTCHA / bot check),
she does not hand it to the owner first. She works a strategy ladder,
memory-ranked:

    retry -> wait_for_proof_of_work -> captcha_service -> fresh_session
    -> owner_takeover (LAST RESORT ONLY)

Every attempt is recorded (strategy, outcome, ms, detail) into
:class:`ChallengeMemory` when one is attached, and — when escalation
does happen — the escalation message lists exactly what she already
tried, so the owner sees the work before the ask. If the wall clears,
``cleared`` is True and ``via`` names the winning strategy; if it does
not, the outcome says so honestly — no fake "solved" claims.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..core.logging_setup import get_logger
from . import errors as _errors
from .challenge_memory import ChallengeMemory, _site_of
from .solved_sessions import (
    SolvedSessionStore,
    restore_solved_session,
    save_solved_session,
)

_log = get_logger(__name__)

__all__ = [
    "AttemptRecord",
    "ChallengeOutcome",
    "ChallengeSignal",
    "ChallengeSolver",
    "DEFAULT_ORDER",
    "STRATEGY_CAPTCHA_SERVICE",
    "STRATEGY_FRESH_SESSION",
    "STRATEGY_RETRY",
    "STRATEGY_TAKEOVER",
    "STRATEGY_WAIT",
    "detect_challenge_signal",
    "format_escalation_message",
]

STRATEGY_RETRY = "retry"
STRATEGY_WAIT = "wait_proof_of_work"
STRATEGY_CAPTCHA_SERVICE = "captcha_service"
STRATEGY_FRESH_SESSION = "fresh_session"
STRATEGY_TAKEOVER = "owner_takeover"

#: Default strategy order when memory has no opinion.
DEFAULT_ORDER = [STRATEGY_RETRY, STRATEGY_WAIT, STRATEGY_CAPTCHA_SERVICE,
                 STRATEGY_FRESH_SESSION, STRATEGY_TAKEOVER]


class TabLike(Protocol):
    """Duck-typed live tab. Anything with these members works."""

    @property
    def url(self) -> str: ...
    @property
    def title(self) -> str: ...
    def html(self) -> str: ...
    def reload(self) -> None: ...


@dataclass
class ChallengeSignal:
    kind: str
    vendor: str
    detection: str
    url: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "vendor": self.vendor,
                "detection": self.detection, "url": self.url,
                "evidence": self.evidence}


@dataclass
class AttemptRecord:
    strategy: str
    ok: bool
    ms: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "ok": self.ok, "ms": self.ms,
                "detail": self.detail}


@dataclass
class ChallengeOutcome:
    cleared: bool
    via: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    takeover_url: str | None = None
    takeover_grant: Any | None = None
    resume_state: dict[str, Any] | None = None
    message: str = ""
    signal: ChallengeSignal | None = None
    #: Cookies persisted to the solved-session store on this clear (0 =
    #: nothing newly saved — e.g. a restore-clear re-saves, a
    #: still-blocked outcome saves nothing).
    session_saved: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"cleared": self.cleared, "via": self.via,
                "attempts": [a.to_dict() for a in self.attempts],
                "takeover_url": self.takeover_url,
                "resume_state": self.resume_state,
                "message": self.message,
                "signal": self.signal.to_dict() if self.signal else None,
                "session_saved": self.session_saved}


def detect_challenge_signal(html: str = "", url: str = "", *,
                            title: str = "", headers: Any = None
                            ) -> ChallengeSignal | None:
    """Detect a wall on a page. ``None`` = no challenge.

    Wall detection reuses the gated :func:`errors.detect_challenge`
    (title family / WAF headers / body markers), then
    :func:`tools.captcha.detect` for CAPTCHA-kind granularity — a page
    with a reCAPTCHA widget but no wall title still counts.
    """
    detection = _errors.detect_challenge(html or "", title=title,
                                         headers=headers)
    if detection:
        from .botwall import vendor_from_detection
        return ChallengeSignal(kind=detection,
                               vendor=vendor_from_detection(detection),
                               detection=detection, url=url,
                               evidence=f"wall marker: {detection}")
    try:
        from ..tools.captcha import detect as _captcha_detect
        found = _captcha_detect(html or "", url or "")
    except Exception:  # noqa: BLE001 - captcha detection never blocks
        found = []
    if found:
        first = found[0]
        return ChallengeSignal(kind=str(getattr(first, "kind", "captcha")),
                               vendor="captcha", detection="captcha-wall",
                               url=url,
                               evidence=f"captcha widget: {first.kind}")
    return None


def format_escalation_message(*, signal_kind: str, url: str,
                              attempts: list[tuple[str, bool, str]] | list[AttemptRecord],
                              takeover_url: str,
                              task_label: str = "") -> str:
    """The owner escalation message. Lists what was already tried.

    The attempt list is the anti-laziness receipt: the owner sees Devon
    worked the ladder before asking for a hand.
    """
    lines = [
        f"🧱 Browser wall: {signal_kind} on {url or 'the page'}",
    ]
    if task_label:
        lines.append(f"task: {task_label}")
    lines.append("I tried these first — none cleared it:")
    for attempt in attempts:
        if isinstance(attempt, AttemptRecord):
            name, ok, detail = attempt.strategy, attempt.ok, attempt.detail
        else:
            name, ok, detail = attempt
        mark = "✓" if ok else "✗"
        lines.append(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    lines.append("")
    lines.append("Your turn — tap to take over this exact browser session:")
    lines.append(takeover_url)
    lines.append("Tap the link to open it right in Telegram, or copy it and "
                 "open it in Chrome if the view feels cramped — same live "
                 "session either way.")
    lines.append("Solve the challenge in the live view, then tap "
                 "\"Done, take over\" — I'll check the wall is gone and "
                 "pick up where I left off. (Or just close it — I'll "
                 "re-check anyway.)")
    return "\n".join(lines)


class ChallengeSolver:
    """Run the strategy ladder against one wall."""

    def __init__(self, *,
                 memory: ChallengeMemory | None = None,
                 service_solve: Callable[[Any], Any] | None = None,
                 fresh_session: Callable[[], Any] | None = None,
                 takeover_manager: Any | None = None,
                 notifier: Callable[[str, str], Any] | None = None,
                 session_store: SolvedSessionStore | None = None,
                 wait_s: float = 3.0,
                 sleeper: Callable[[float], None] | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self._memory = memory
        self._service_solve = service_solve
        self._fresh_session = fresh_session
        self._takeover_manager = takeover_manager
        self._notifier = notifier
        self._session_store = session_store
        self._wait_s = max(0.0, float(wait_s))
        self._sleep = sleeper or time.sleep
        self._clock = clock or time.time

    # -- main entry ------------------------------------------------------------
    def solve(self, *, tab: TabLike | None = None, html: str = "",
              url: str = "", title: str = "", headers: Any = None,
              resume_state: dict[str, Any] | None = None,
              owner_chat: str = "", task_label: str = "",
              session_id: str = "") -> ChallengeOutcome:
        """Detect + work the ladder. Takeover is the last rung only."""
        started = self._clock()
        page_html = tab.html() if tab is not None else html
        page_url = tab.url if tab is not None else url
        page_title = tab.title if tab is not None else title
        signal = detect_challenge_signal(page_html, page_url,
                                         title=page_title, headers=headers)
        if signal is None:
            return ChallengeOutcome(cleared=True, via="no_challenge",
                                    message="no wall on this page")
        # A previous visit may already have cleared this wall: reuse the
        # solved session before spending the ladder on it. A restore
        # that does not clear falls through to the ladder.
        if self._session_store is not None and tab is not None:
            restored = restore_solved_session(
                tab, page_url, self._session_store)
            if restored and self._recheck(tab, page_html):
                saved = self._save_solved(tab, page_url, "saved_session")
                self._record_memory(signal, [AttemptRecord(
                    strategy="saved_session", ok=True,
                    detail=f"restored {restored} cookies")],
                    cleared=True, via="saved_session",
                    takeover_needed=False, started=started)
                return ChallengeOutcome(
                    cleared=True, via="saved_session",
                    attempts=[AttemptRecord(
                        strategy="saved_session", ok=True,
                        detail=f"restored {restored} cookies")],
                    resume_state=resume_state, signal=signal,
                    message=f"wall already cleared — reused {restored} "
                            f"saved cookies",
                    session_saved=saved)
        attempts: list[AttemptRecord] = []
        order = self._ordered(signal)
        cleared_via = ""
        for strategy in order:
            if strategy == STRATEGY_TAKEOVER:
                break  # handled below, after every self-solve rung
            record = self._attempt(strategy, tab=tab, signal=signal)
            attempts.append(record)
            if self._recheck(tab, page_html):
                cleared_via = strategy
                break
        if cleared_via:
            saved = self._save_solved(tab, page_url, cleared_via)
            self._record_memory(signal, attempts, cleared=True,
                                via=cleared_via, takeover_needed=False,
                                started=started)
            return ChallengeOutcome(
                cleared=True, via=cleared_via, attempts=attempts,
                resume_state=resume_state, signal=signal,
                message=f"wall cleared via {cleared_via}",
                session_saved=saved)
        # Ladder exhausted — LAST RESORT: owner takeover.
        attempts.append(AttemptRecord(
            strategy=STRATEGY_TAKEOVER, ok=False, detail="escalating"))
        if self._takeover_manager is None:
            self._record_memory(signal, attempts, cleared=False,
                                via="takeover_required",
                                takeover_needed=True, started=started)
            return ChallengeOutcome(
                cleared=False, via="takeover_required", attempts=attempts,
                resume_state=resume_state, signal=signal,
                message="wall persists and no takeover manager is "
                        "configured — cannot escalate")
        try:
            grant = self._takeover_manager.request_takeover(
                session_id=session_id or f"challenge-{int(started * 1000)}",
                url=page_url, task_label=task_label or "browsing",
                signal_kind=signal.kind, attempts=attempts,
                resume_state=resume_state or {}, tab=tab,
                owner_chat=owner_chat,
            )
        except Exception as exc:  # noqa: BLE001 - e.g. stream bind failure
            # An unwound exception here would skip the memory record and
            # surface as a raw "wall handler failed" — the encounter is
            # still data, and the owner deserves an honest outcome, not
            # a traceback.
            self._record_memory(signal, attempts, cleared=False,
                                via="takeover_failed",
                                takeover_needed=True, started=started)
            return ChallengeOutcome(
                cleared=False, via="takeover_failed", attempts=attempts,
                resume_state=resume_state, signal=signal,
                message="wall persists and takeover could not start: "
                        f"{exc}")
        message = format_escalation_message(
            signal_kind=signal.kind, url=page_url, attempts=attempts,
            takeover_url=grant.url, task_label=task_label or "browsing")
        if grant.needs_public_base:
            # The link is host-network-only until the operator sets a
            # public base URL — say so in the escalation itself instead
            # of sending a link that silently fails on the owner's phone.
            message += (
                "\n\n⚠️ No public takeover URL is configured "
                "(NM_TAKEOVER_BASE_URL) — this link works on the host "
                "network only until a tunnel / port-forward is set up.")
        if self._notifier is not None:
            try:
                self._notifier(message, grant.url)
            except Exception:  # noqa: BLE001 - notify failure is reported
                _log.warning("challenge escalation notify failed",
                             exc_info=True)
        self._record_memory(signal, attempts, cleared=False,
                            via="takeover_pending", takeover_needed=True,
                            started=started)
        return ChallengeOutcome(
            cleared=False, via="takeover_pending", attempts=attempts,
            takeover_url=grant.url, takeover_grant=grant,
            resume_state=resume_state, message=message, signal=signal)

    # -- ladder ------------------------------------------------------------------
    def _ordered(self, signal: ChallengeSignal) -> list[str]:
        if self._memory is not None:
            try:
                return self._memory.rank_strategies(
                    _site_of(signal.url), signal.kind,
                    list(DEFAULT_ORDER))
            except Exception:  # noqa: BLE001 - memory never blocks solving
                _log.debug("challenge memory rank failed", exc_info=True)
        return list(DEFAULT_ORDER)

    def _attempt(self, strategy: str, *, tab: TabLike | None,
                 signal: ChallengeSignal) -> AttemptRecord:
        started = self._clock()

        def done(ok: bool, detail: str) -> AttemptRecord:
            return AttemptRecord(strategy=strategy, ok=ok,
                                 ms=int((self._clock() - started) * 1000),
                                 detail=detail)

        try:
            if strategy == STRATEGY_RETRY:
                if tab is None:
                    return done(False, "no tab to reload")
                tab.reload()
                return done(True, "reloaded")
            if strategy == STRATEGY_WAIT:
                # FriendlyCaptcha-style proof-of-work solves itself given
                # time; for other walls the wait is just a settle pause.
                self._sleep(self._wait_s)
                return done(True, f"waited {self._wait_s:g}s")
            if strategy == STRATEGY_CAPTCHA_SERVICE:
                if self._service_solve is None:
                    return done(False, "no solving service configured")
                from ..tools.captcha import detect as _captcha_detect
                html = tab.html() if tab is not None else ""
                challenges = _captcha_detect(html, signal.url)
                if not challenges:
                    return done(False, "no captcha widget to solve")
                result = self._service_solve(challenges)
                if getattr(result, "ok", False):
                    return done(True, "solving service returned a token")
                return done(False, "solving service failed")
            if strategy == STRATEGY_FRESH_SESSION:
                if self._fresh_session is None:
                    return done(False, "no fresh-session callback")
                self._fresh_session()
                return done(True, "fresh session started")
            return done(False, f"unknown strategy {strategy!r}")
        except Exception as exc:  # noqa: BLE001 - attempt failure is data
            return done(False, f"{type(exc).__name__}: {exc}")

    def _recheck(self, tab: TabLike | None, old_html: str) -> bool:
        """True when the wall is gone. A recheck that errors is NOT a clear."""
        if tab is None:
            return False
        try:
            html = tab.html()
        except Exception:  # noqa: BLE001 - unreadable page is not cleared
            return False
        return detect_challenge_signal(html, tab.url,
                                       title=tab.title) is None

    def _save_solved(self, tab: TabLike | None, url: str, via: str, *,
                     grant: Any | None = None) -> int:
        """Persist a cleared tab's cookies + storage. 0 = nothing saved.

        Called ONLY on a verified clear — never on a still-blocked
        outcome. Best effort: a save failure never breaks the resume.
        With a takeover grant, the save is audited (site + count +
        via, never cookie values).
        """
        if self._session_store is None or tab is None:
            return 0
        count = save_solved_session(tab, url, self._session_store,
                                    cleared_via=via)
        if count and grant is not None \
                and self._takeover_manager is not None:
            try:
                self._takeover_manager.record_session_saved(
                    grant, site=url, cookie_count=count, via=via)
            except Exception:  # noqa: BLE001 - audit never blocks resume
                _log.debug("session_saved audit failed", exc_info=True)
        return count

    def _record_memory(self, signal: ChallengeSignal,
                       attempts: list[AttemptRecord], *, cleared: bool,
                       via: str, takeover_needed: bool,
                       started: float) -> None:
        if self._memory is None:
            return
        try:
            self._memory.record(
                site=signal.url, kind=signal.kind, vendor=signal.vendor,
                strategies=[(a.strategy, a.ok, a.ms) for a in attempts],
                cleared=cleared, via=via,
                takeover_needed=takeover_needed,
                ms=int((self._clock() - started) * 1000),
                started_at=started)
        except Exception:  # noqa: BLE001 - memory never blocks solving
            _log.debug("challenge memory record failed", exc_info=True)

    # -- AgentLoop hook ----------------------------------------------------------
    def wall_handler_for(self, tab: TabLike | None = None, *,
                         resume_state: dict[str, Any] | None = None,
                         owner_chat: str = "",
                         task_label: str = "",
                         on_outcome: Callable[[str, dict[str, Any]], None]
                         | None = None) -> Callable[[Any, str], str]:
        """An ``AgentLoop.on_blocked`` callback: solve or stop honestly.

        Returns ``"continue"`` when the wall cleared (self-solved, or
        takeover resolved), anything else stops the loop.

        ``on_outcome`` (optional) reports wall lifecycle events —
        ``wall_hit`` / ``wall_escalated`` / ``wall_cleared`` /
        ``wall_blocked`` with ``{"detection", "via", ...}`` — so a
        background job can notify the owner and record the wall
        outcome. It never affects solving: callback errors are logged,
        not raised.
        """
        def _emit(kind: str, info: dict[str, Any]) -> None:
            if on_outcome is None:
                return
            try:
                on_outcome(kind, info)
            except Exception:  # noqa: BLE001 - reporting never breaks solving
                _log.debug("wall on_outcome callback failed", exc_info=True)

        def _handler(observation: Any, detection: str) -> str:
            # Resume state is fresh at call time: the loop's current
            # step + page URL, merged over the base state the caller
            # passed. A static captured state would resume the loop
            # where the handler was BUILT, not where the wall hit.
            page = getattr(observation, "page", None)
            _emit("wall_hit", {"detection": detection,
                               "url": getattr(page, "url", "") or ""})
            fresh_resume = dict(resume_state or {})
            fresh_resume["step"] = getattr(observation, "step", None)
            fresh_resume["url"] = getattr(page, "url", "") or \
                fresh_resume.get("url", "")
            outcome = self.solve(
                tab=tab, html=getattr(page, "html", ""),
                url=getattr(page, "url", ""),
                title=getattr(page, "title", ""),
                resume_state=fresh_resume, owner_chat=owner_chat,
                task_label=task_label)
            if outcome.cleared:
                _emit("wall_cleared", {"detection": detection,
                                       "via": outcome.via})
                return "continue"
            if outcome.takeover_grant is not None \
                    and self._takeover_manager is not None:
                _emit("wall_escalated", {
                    "detection": detection, "via": "takeover_pending",
                    "takeover_url": outcome.takeover_url or ""})
                res = self._takeover_manager.wait_for_resolution(
                    outcome.takeover_grant,
                    cleared_check=lambda: self._recheck(tab, ""),
                    timeout_s=getattr(
                        self._takeover_manager, "wait_timeout_s", 900),
                    poll_s=2.0)
                if res.cleared:
                    # Owner cleared it — persist the solved session so
                    # the next visit reuses it. Still-blocked saves
                    # nothing: _save_solved is only called on a clear.
                    page_url = getattr(
                        getattr(observation, "page", None), "url", "") or (
                            tab.url if tab is not None else "")
                    self._save_solved(tab, page_url, f"takeover:{res.via}",
                                      grant=outcome.takeover_grant)
                    _emit("wall_cleared", {
                        "detection": detection,
                        "via": f"takeover:{res.via}"})
                    return "continue"
                _emit("wall_blocked", {
                    "detection": detection,
                    "via": "takeover:still_blocked"})
                return "stop"
            _emit("wall_blocked", {"detection": detection,
                                   "via": outcome.via})
            return "stop"
        return _handler
