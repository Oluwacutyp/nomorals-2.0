"""Mission progress: chat-visible status, milestone pushes, stall tracking.

This module is the chat-visible face of missions. It answers three questions
the owner asks in chat:

1. **How far along is it?** — ``MissionStore.detail`` + ``render_status_text``
   (% complete, current step, an honest ETA, budget burn).
2. **Tell me when something happens.** — ``MissionMilestones`` pushes
   started / step / stalled / done milestones through the existing
   :class:`nomorals.agents.notifier.Notifier`. No parallel notifier: the
   Wave D choke point (dedupe, feature flag, quiet-hours holds, persisted
   delivery states) applies to every mission push.
3. **Why is it stuck?** — stalled-reason tracking. A stalled mission
   records a *concrete* blocker (``StallCode`` + message + step + since),
   never a bare "waiting". The runner records these automatically
   (consecutive-failure budget, wall/token budget exhaustion) and exposes
   ``mark_stalled`` / ``clear_stalled`` for the concrete external reasons
   only an operator knows: "waiting on provider X", "blocked on approval Y",
   "dependency Z missing".

Anti-spam policy (the Wave D discipline, applied to missions):

- ``started`` / ``stalled`` / ``terminal`` (done/failed/cancelled) pushes are
  *transition*-gated: each fires once per mission, the marker persisted in
  ``mission.state["milestones"]`` so a restart/resume never re-sends.
- ``step`` pushes are *cooldown*-gated (default 600s, the same window as the
  Notifier dedupe) and *quiet-hours*-held: a step update during quiet hours
  is stored as ``held-quiet-hours`` (visible in ``/notify``, redeliverable by
  the existing ``Notifier.redeliver``) instead of waking the owner with a
  transient "%" note.
- The Notifier's own (kind, title) dedupe is the backstop for all of it.

Every method on ``MissionMilestones`` is telemetry-safe: it never raises.
A milestone push must not be able to break a mission run.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from ..agents.notifier import (
    STATE_HELD_QUIET,
    Notifier,
    _in_quiet_hours_now,
)
from ..core.errors import ValidationError
from ..core.logging_setup import get_logger
from ..storage.kv import KVStore

__all__ = [
    "StallCode",
    "clear_stall",
    "estimate_eta",
    "fmt_duration",
    "MissionMilestones",
    "MissionWatchers",
    "real_plan_steps",
    "record_stall",
    "render_status_text",
]

_log = get_logger(__name__)

#: after this many consecutive step failures the runner declares the
#: mission stalled instead of silently grinding through the plan.
STALL_AFTER_FAILURES = 3


class StallCode:
    """Concrete reasons a mission can be stuck. Never a bare 'waiting'."""

    WAITING_ON_PROVIDER = "waiting_on_provider"
    BLOCKED_ON_APPROVAL = "blocked_on_approval"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEPENDENCY_MISSING = "dependency_missing"

    ALL = frozenset({
        WAITING_ON_PROVIDER,
        BLOCKED_ON_APPROVAL,
        RETRY_BUDGET_EXHAUSTED,
        BUDGET_EXHAUSTED,
        DEPENDENCY_MISSING,
    })

    #: human-readable one-liners for the status command.
    LABELS = {
        WAITING_ON_PROVIDER: "waiting on provider",
        BLOCKED_ON_APPROVAL: "blocked on approval",
        RETRY_BUDGET_EXHAUSTED: "retry budget exhausted",
        BUDGET_EXHAUSTED: "budget exhausted",
        DEPENDENCY_MISSING: "dependency missing",
    }

    #: what would unblock the mission for each stall code. The status
    #: command and the stall reply always ship these, so a stall never
    #: reads as a bare "stalled" with no way out.
    UNBLOCK_HINTS = {
        WAITING_ON_PROVIDER: "unblocks when the provider recovers, "
            "or switch providers and /mission clear it",
        BLOCKED_ON_APPROVAL: "unblocks when the owner approves — reply to "
            "the approval request, then /mission clear it",
        RETRY_BUDGET_EXHAUSTED: "unblocks with /mission clear after fixing "
            "the failing step, or /mission retry for a fresh attempt",
        BUDGET_EXHAUSTED: "unblocks when the budget is raised, or "
            "/mission retry for a fresh attempt with a bigger budget",
        DEPENDENCY_MISSING: "unblocks when the missing dependency is "
            "installed, then /mission clear it",
    }


# ── stall state (pure helpers on the Mission object) ─────────────────────────

_STALL_KEY = "stall"


def record_stall(
    mission: Any,
    code: str,
    message: str,
    *,
    step: str = "",
    extra: dict[str, Any] | None = None,
) -> bool:
    """Record a concrete stall reason on the mission's state.

    Returns ``True`` when this is a *new or changed* stall (a push-worthy
    transition); ``False`` when the identical stall is already recorded, so
    callers can avoid re-notifying.  Raises :class:`ValidationError` for an
    unknown code — fail fast, no silent generic strings.
    """
    if code not in StallCode.ALL:
        raise ValidationError(
            f"unknown stall code {code!r} — one of: {sorted(StallCode.ALL)}",
            field="code",
        )
    if not message or not message.strip():
        raise ValidationError("a stall reason needs a message", field="message")
    prev = mission.state.get(_STALL_KEY) or {}
    mission.state[_STALL_KEY] = {
        "code": code,
        "message": message.strip(),
        "step": step or "",
        "since": time.time(),
        "extra": dict(extra or {}),
    }
    return prev.get("code") != code or prev.get("message") != message.strip()


def clear_stall(mission: Any) -> bool:
    """Drop the stall record (called when the mission makes progress again)."""
    return mission.state.pop(_STALL_KEY, None) is not None


# ── plan / ETA ───────────────────────────────────────────────────────────────

def _step_name(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("name") or "")
    return str(entry)


def real_plan_steps(plan: Any) -> list[Any]:
    """Plan entries minus the persisted ``__plan_error__`` degradation marker."""
    return [
        s for s in (plan or [])
        if not (isinstance(s, dict) and "__plan_error__" in s)
    ]


def fmt_duration(seconds: float | None) -> str:
    """'3661' -> '1h 1m'. Never raises; None -> 'unknown'."""
    if seconds is None:
        return "unknown"
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "unknown"
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def estimate_eta(mission: Any) -> tuple[float | None, str]:
    """Honest ETA from measured per-step wall time.

    Returns ``(seconds, note)``. ``seconds`` is ``None`` when there is not
    enough measured data — the note then says *why* ("no step timing yet",
    "no plan stored") instead of inventing a number. When the wall budget
    would run out first, the note says so explicitly.
    """
    steps = real_plan_steps(mission.state.get("plan"))
    if not steps:
        return None, "no plan stored yet"
    completed = set(mission.state.get("completed_steps") or [])
    done = sum(1 for s in steps if _step_name(s) in completed)
    remaining = len(steps) - done
    if remaining <= 0:
        return 0.0, "all steps complete"
    if done <= 0 or mission.spent_wall <= 0:
        return None, "no step timing yet"
    per_step = mission.spent_wall / done
    eta = per_step * remaining
    note = ""
    budget_wall = float(mission.budget_wall or 0.0)
    if budget_wall:
        wall_left = max(0.0, budget_wall - mission.spent_wall)
        if wall_left < eta:
            note = f"wall budget runs out first (~{fmt_duration(wall_left)} left)"
    return eta, note


# ── chat rendering ───────────────────────────────────────────────────────────

def render_status_text(detail: dict[str, Any]) -> str:
    """One chat-visible block: % complete, current step, ETA, stall reason.

    Takes the dict from ``MissionStore.detail``. Every line is real,
    persisted state — nothing is guessed.
    """
    m = detail.get("mission") or {}
    p = detail.get("progress") or {}
    stall = detail.get("stall")
    lines = [f"🎯 {m.get('name') or 'mission'} [{m.get('status') or '?'}]"]
    goal = str(m.get("goal") or "")
    if goal:
        lines.append(f"goal: {goal[:160]}")
    pct = float(p.get("percent") or 0.0)
    progress_line = (
        f"progress: {p.get('steps_done', 0)}/{p.get('total_steps', 0)} "
        f"steps ({pct:.0f}%)"
    )
    if p.get("current_step"):
        progress_line += f" — current: {p['current_step']}"
    lines.append(progress_line)
    eta = detail.get("eta_seconds")
    eta_note = str(detail.get("eta_note") or "")
    if eta is None:
        lines.append(f"eta: unknown" + (f" — {eta_note}" if eta_note else ""))
    else:
        lines.append(f"eta: ~{fmt_duration(eta)}" + (f" ({eta_note})" if eta_note else ""))
    spent = f"spent: {float(m.get('spent_wall') or 0.0):.0f}s wall, {int(m.get('spent_tokens') or 0)} tokens"
    budget_wall = float(m.get("budget_wall") or 0.0)
    budget_tokens = int(m.get("budget_tokens") or 0)
    if budget_wall or budget_tokens:
        spent += f" (budget: {fmt_duration(budget_wall) if budget_wall else '∞'} / {budget_tokens if budget_tokens else '∞'} tokens)"
    lines.append(spent + f" — {int(m.get('iterations') or 0)} iteration(s)")
    if stall:
        code = stall.get("code")
        label = StallCode.LABELS.get(code, code)
        stall_line = f"⚠️ stalled [{code}]: {label} — {stall.get('message')}"
        if stall.get("step"):
            stall_line += f" (step: {stall['step']})"
        since = stall.get("since")
        if since:
            stall_line += f" [since {time.strftime('%H:%M', time.localtime(since))}]"
        lines.append(stall_line)
        hint = StallCode.UNBLOCK_HINTS.get(code)
        if hint:
            lines.append(f"   → unblocks: {hint}")
    elif str(m.get("status") or "") in {"running", "paused", "pending"} \
            and not p.get("last_error"):
        lines.append("state: making progress")
    if p.get("last_error"):
        lines.append(f"last error: {str(p['last_error'])[:160]}")
    return "\n".join(ln for ln in lines if ln)


# ── milestone pushes ─────────────────────────────────────────────────────────

class MissionWatchers:
    """Chat-key subscriptions for mission milestone pushes.

    ``/mission watch <id>`` subscribes the current chat; ``MissionMilestones``
    fans every milestone push (started / step / stalled / terminal) out to
    the subscribed chats on top of the normal Notifier publish to the owner
    channels. Subscriptions persist in ``kv_store`` as JSON::

        mission.watch.<mission_id> -> ["telegram:12345", "console:main"]

    All methods are telemetry-safe: with no db they are silent no-ops, and
    every db touch is best-effort so a subscription failure never breaks a
    mission run or a chat command.
    """

    PREFIX = "mission.watch."

    def __init__(self, db: Any) -> None:
        self.db = db

    def _key(self, mission_id: str) -> str:
        return f"{self.PREFIX}{mission_id}"

    def _read(self, mission_id: str) -> list[str]:
        if self.db is None:
            return []
        try:
            data = KVStore(self.db).get(self._key(mission_id))
        except Exception:  # noqa: BLE001 - subscriptions are best-effort
            return []
        if not data:
            return []
        return [str(c) for c in data if c] if isinstance(data, list) else []

    def _write(self, mission_id: str, chats: list[str]) -> None:
        if self.db is None:
            return
        try:
            KVStore(self.db).set(self._key(mission_id), chats)
        except Exception:  # noqa: BLE001 - subscriptions are best-effort
            pass

    def subscribe(self, mission_id: str, chat_key: str) -> bool:
        """Subscribe a chat; returns True when it was newly added."""
        chats = self._read(mission_id)
        if chat_key in chats:
            return False
        chats.append(chat_key)
        self._write(mission_id, chats)
        return True

    def unsubscribe(self, mission_id: str, chat_key: str) -> bool:
        """Unsubscribe a chat; returns True when one was removed."""
        chats = self._read(mission_id)
        if chat_key not in chats:
            return False
        chats = [c for c in chats if c != chat_key]
        self._write(mission_id, chats)
        return True

    def watchers(self, mission_id: str) -> list[str]:
        """Chat keys subscribed to this mission's milestones."""
        return self._read(mission_id)


class MissionMilestones:
    """Proactive chat pushes for mission milestones, via the Notifier.

    Events: ``started`` (once), ``step`` (cooldown-gated progress),
    ``stalled`` (on transition), ``terminal`` (done/failed/cancelled, once
    each). Every push funnels through ``Notifier.publish(kind="mission")``
    so the Wave D choke point — dedupe, the ``notifier`` feature flag, and
    persisted delivery states — governs mission pushes exactly like every
    other alert. Nothing here may raise: milestone reporting is telemetry,
    and telemetry never breaks a run.
    """

    KIND = "mission"
    MILESTONE_LOG_KEY = "milestones"
    LOG_LIMIT = 30

    def __init__(
        self,
        context: Any,
        *,
        store: Any = None,
        clock: Callable[[], float] | None = None,
        step_cooldown_seconds: float = 600.0,
        watch_store: "MissionWatchers | None" = None,
    ) -> None:
        self.context = context
        if store is None:
            from .mission import MissionStore

            store = MissionStore(context.db)
        self.store = store
        self._clock = clock or time.time
        self.step_cooldown = max(0.0, float(step_cooldown_seconds))
        self._notifier = Notifier(context)
        # Chat-key subscriptions fanned out on top of the Notifier publish.
        # Default: backed by the context db (silent no-op when db is None).
        if watch_store is None:
            watch_store = MissionWatchers(getattr(context, "db", None))
        self.watch_store = watch_store

    # ── events ───────────────────────────────────────────────────────────────

    def _fanout(self, mission: Any, title: str, body: str) -> None:
        """Push the same milestone to subscribed chats (``/mission watch``).

        This rides on top of the normal Notifier publish, which still goes
        to the owner channels with all Wave D discipline intact. Chats that
        are already owner channels are skipped (the Notifier reached them).
        Never raises: fan-out is telemetry.
        """
        try:
            store = self.watch_store
            if store is None:
                return
            chats = store.watchers(getattr(mission, "id", ""))
        except Exception:  # noqa: BLE001 - telemetry never breaks a run
            return
        if not chats:
            return
        gateway = getattr(self._notifier, "gateway", None)
        if gateway is None:
            return
        from ..social.chat.base import ChatRef

        owner_keys = set()
        settings = getattr(self.context, "settings", None)
        partner = getattr(settings, "partner", None)
        raw = getattr(partner, "owner_chats", "") or ""
        for key in str(raw).split(","):
            key = key.strip()
            if key:
                owner_keys.add(key)
        text = f"🔔 {title}" + (f"\n{body}" if body else "")
        text = text[:3900]
        for key in chats:
            plat, _, cid = str(key).partition(":")
            if not (plat and cid) or key in owner_keys:
                continue
            try:
                gateway.send(plat, ChatRef(platform=plat, chat_id=cid), text)
            except Exception:  # noqa: BLE001 - one dead chat must not kill the rest
                continue

    def on_started(self, mission: Any) -> dict[str, Any]:
        """Push once when a mission starts running. Resume-safe."""
        try:
            if self._marked(mission, "started"):
                return {"suppressed": "already-notified", "event": "started"}
            title = f"mission started: {mission.name}"
            body = (
                f"goal: {mission.goal[:200]}\n"
                + self._budget_line(mission)
            )
            res = self._notifier.publish(self.KIND, title, body)
            self._fanout(mission, title, body)
            self._mark(mission, "started", title)
            # a fresh start push counts against the step cooldown, so the
            # first completing step doesn't double-notify seconds later.
            mission.state["last_progress_push"] = self._clock()
            self.store.save(mission)
            return res
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("mission milestone started failed: %s", exc)
            return {"suppressed": "error", "event": "started"}

    def on_step(self, mission: Any, outcome: Any) -> dict[str, Any]:
        """Cooldown-gated progress push for a completed step."""
        try:
            return self._on_step(mission, outcome)
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("mission milestone step failed: %s", exc)
            return {"suppressed": "error", "event": "step"}

    def _on_step(self, mission: Any, outcome: Any) -> dict[str, Any]:
        now = self._clock()
        last = float(mission.state.get("last_progress_push") or 0.0)
        if now - last < self.step_cooldown:
            return {
                "suppressed": "cooldown",
                "event": "step",
                "cooldown_remaining": round(self.step_cooldown - (now - last), 1),
            }
        detail = self.store.detail(mission.id)
        progress = detail.get("progress") or {}
        pct = float(progress.get("percent") or 0.0)
        title = f"mission update: {mission.name} — {pct:.0f}%"
        body = render_status_text(detail)
        if _in_quiet_hours_now(self.context):
            # Non-critical progress note during quiet hours: held, not
            # sent. Stored as held-quiet-hours so /notify shows it and the
            # existing Notifier.redeliver() can send it when quiet ends.
            res = self._notifier._store(  # noqa: SLF001 - deliberate: held send
                self.KIND, title, body, 0, delivery_state=STATE_HELD_QUIET
            )
            res["held"] = "quiet-hours"
        else:
            res = self._notifier.publish(self.KIND, title, body)
        self._fanout(mission, title, body)
        mission.state["last_progress_push"] = now
        self.store.save(mission)
        return res

    def on_stalled(self, mission: Any) -> dict[str, Any]:
        """Push immediately when a *new* stall reason is recorded."""
        try:
            stall = mission.state.get(_STALL_KEY) or {}
            if not stall:
                return {"suppressed": "no-stall", "event": "stalled"}
            key = f"stalled:{stall.get('code')}"
            if self._marked(mission, key):
                return {"suppressed": "already-notified", "event": "stalled"}
            # the stall code rides in the title so the Notifier's
            # (kind, title) dedupe treats a *different* blocker as a new
            # alert while collapsing repeats of the same one.
            title = f"mission stalled [{stall.get('code')}]: {mission.name}"
            body = render_status_text(self.store.detail(mission.id))
            res = self._notifier.publish(self.KIND, title, body)
            self._fanout(mission, title, body)
            self._mark(mission, key, title)
            return res
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("mission milestone stalled failed: %s", exc)
            return {"suppressed": "error", "event": "stalled"}

    def on_terminal(self, mission: Any, status: str, *, error: str = "") -> dict[str, Any]:
        """Push once per terminal transition (done / failed / cancelled)."""
        try:
            key = f"terminal:{status}"
            if self._marked(mission, key):
                return {"suppressed": "already-notified", "event": "terminal"}
            word = {"done": "done ✅", "failed": "failed ❌",
                    "cancelled": "cancelled"}.get(status, status)
            title = f"mission {word}: {mission.name}"
            detail = self.store.detail(mission.id)
            body = render_status_text(detail)
            if error:
                body += f"\nerror: {error[:200]}"
            res = self._notifier.publish(self.KIND, title, body)
            self._fanout(mission, title, body)
            self._mark(mission, key, title)
            return res
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("mission milestone terminal failed: %s", exc)
            return {"suppressed": "error", "event": "terminal"}

    # ── milestone log (persisted markers: resume-safe, process-safe) ─────────

    def _marked(self, mission: Any, key: str) -> bool:
        log = mission.state.get(self.MILESTONE_LOG_KEY) or []
        return any(isinstance(e, dict) and e.get("event") == key for e in log)

    def _mark(self, mission: Any, key: str, title: str) -> None:
        log = list(mission.state.get(self.MILESTONE_LOG_KEY) or [])
        log.append({"event": key, "title": title, "at": self._clock()})
        mission.state[self.MILESTONE_LOG_KEY] = log[-self.LOG_LIMIT:]
        self.store.save(mission)

    @staticmethod
    def _budget_line(mission: Any) -> str:
        wall = float(getattr(mission, "budget_wall", 0.0) or 0.0)
        tokens = int(getattr(mission, "budget_tokens", 0) or 0)
        if not wall and not tokens:
            return "budget: unlimited"
        return (f"budget: {fmt_duration(wall) if wall else '∞'} wall / "
                f"{tokens if tokens else '∞'} tokens")
