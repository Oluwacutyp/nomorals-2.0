"""The autonomy agent: she initiates, on her own clock.

A real partner doesn't wait to be asked. This agent runs a heartbeat and,
from her *current mood*, the time of day, how long it's been quiet, and the
facts of her life (background pack), decides whether to:

* send the partner a spontaneous DM (check-in, miss-you, sharing her day), or
* post to one of the configured group chats, on a topic that fits her
  interests — never about the partner, never personal.

Modes (``partner.autonomy_mode``):

* ``off``     — the agent does not start.
* ``suggest`` — every intended send becomes a *proposal* in ``proactive_log``
  (status=pending). Nothing leaves the machine until the owner approves it
  (``nm partner --approve <id>``). This is the default and the safe one.
* ``auto``    — sends go out directly, subject to quiet hours, per-day caps,
  per-chat minimum intervals, and the gateway's hourly window. Everything is
  still journaled in ``proactive_log`` and the audit trail.  A cap of ``0``
  means *unlimited* (the owner accepts the volume).

Rules this agent will not break, in any mode:

* No sends during quiet hours.
* At most ``max_dm_per_day`` proactive DMs and ``max_group_per_day`` group
  posts per day (0 = unlimited).
* At most one proactive send per chat per ``min_interval_minutes``.
* Group content never mentions the partner, the relationship, or that she
  is an AI.
* If the model call fails, she says nothing. Proactive silence is always
  the better failure mode than a broken, off-model message.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import ulid_now
from ..core.logging_setup import get_logger
from ..llm.base import Message, SamplingParams
from ..partner.mood import MoodEngine
from ..partner.persona import Persona
from ..partner.relationship import Relationship
from ..partner.style import clamp_to_budget, split_messages, strip_robotic
from ..social.chat.base import ChatKind, ChatRef
from ..social.chat.gateway import ChatGateway

__all__ = ["AutonomyAgent"]

_log = get_logger(__name__)

MIN_INTERVAL_MINUTES = 45


@dataclass
class _DayState:
    day: str = ""
    dm_sent: int = 0
    group_sent: int = 0
    last_dm_ts: float = 0.0
    last_group_ts: float = 0.0
    last_chat_ts: dict[str, float] = field(default_factory=dict)


class AutonomyAgent:
    """Heartbeat-driven proactive messaging, with an approval flow."""

    def __init__(
        self,
        context: Any,
        brain: Any,
        gateway: ChatGateway,
        *,
        mode: str = "suggest",
        owner_chats: set[str] | None = None,
        group_chats: set[str] | None = None,
        quiet_start: int = 22,
        quiet_end: int = 8,
        max_dm_per_day: int = 6,
        max_group_per_day: int = 2,
        heartbeat_seconds: float = 900.0,
        silence_hours_for_checkin: float = 6.0,
    ) -> None:
        if mode not in {"off", "suggest", "auto"}:
            raise ValueError(f"autonomy mode must be off|suggest|auto, got {mode!r}")
        self.context = context
        self.brain = brain
        self.gateway = gateway
        self.mode = mode
        self.owner_chats = set(owner_chats or ())
        self.group_chats = set(group_chats or ())
        self.quiet_start = int(quiet_start)
        self.quiet_end = int(quiet_end)
        # 0 (or less) = unlimited; the owner explicitly asked for no volume caps.
        self.max_dm_per_day = max(0, int(max_dm_per_day))
        self.max_group_per_day = max(0, int(max_group_per_day))
        self.heartbeat_seconds = max(60.0, float(heartbeat_seconds))
        self.silence_hours = max(1.0, float(silence_hours_for_checkin))

        self.mood: MoodEngine = brain.mood
        self.persona: Persona = brain.persona
        self.relationship: Relationship = brain.relationship
        self.background = brain.background
        self._day = _DayState()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.stats = {"ticks": 0, "proposals": 0, "sent": 0, "skipped": 0}

    # ── time rules ───────────────────────────────────────────────────────────
    def _in_quiet_hours(self, now: float) -> bool:
        hour = time.localtime(now).tm_hour
        if self.quiet_start == self.quiet_end:
            return False
        if self.quiet_start > self.quiet_end:  # e.g. 22 -> 8
            return hour >= self.quiet_start or hour < self.quiet_end
        return self.quiet_start <= hour < self.quiet_end

    def _roll_day(self, now: float) -> None:
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if self._day.day != day:
            self._day = _DayState(day=day)

    def _chat_allowed(self, chat_key: str, now: float) -> bool:
        last = self._day.last_chat_ts.get(chat_key, 0.0)
        return (now - last) >= MIN_INTERVAL_MINUTES * 60.0

    @staticmethod
    def _cap_reached(sent: int, cap: int) -> bool:
        """True when a daily counter has hit its cap. ``cap <= 0`` = unlimited."""
        return cap > 0 and sent >= cap

    # ── heartbeat ───────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self.mode == "off" or self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="partner-autonomy", daemon=True)
        self._thread.start()
        _log.info("autonomy agent started (mode=%s)", self.mode)
        return True

    def stop(self) -> None:
        self._stop.set()

    def set_mode(self, mode: str) -> dict[str, Any]:
        """Live mode switch (control command / power mode). ``auto`` starts
        the heartbeat if it isn't running; ``off`` stops it."""
        mode = (mode or "").strip().lower()
        if mode not in {"off", "suggest", "auto"}:
            return {"ok": False, "error": "mode must be off|suggest|auto"}
        self.mode = mode
        if mode == "auto" and not (self._thread is not None and self._thread.is_alive()):
            self.start()
        if mode == "off":
            self.stop()
        _log.info("autonomy mode -> %s", mode)
        return {"ok": True, "mode": mode}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.heartbeat_seconds)
            if self._stop.is_set():
                return
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - one bad tick must not kill the agent
                _log.exception("autonomy tick failed: %s", exc)

    # ── one decision cycle ───────────────────────────────────────────────────
    def tick(self, *, now: float | None = None) -> dict[str, Any]:
        from .features import feature_enabled

        now = now if now is not None else time.time()
        with self._lock:
            self.stats["ticks"] += 1
            if self._in_quiet_hours(now):
                self.stats["skipped"] += 1
                return {"decision": "quiet hours"}
            self._roll_day(now)
            dm_on = feature_enabled(self.context, "proactive_dm")
            groups_on = feature_enabled(self.context, "group_posts")

            # 1) Spontaneous DM to the partner.
            dm_chat = self._owner_dm_chat()
            if dm_chat is not None and dm_on \
                    and not self._cap_reached(self._day.dm_sent, self.max_dm_per_day) \
                    and self._chat_allowed(dm_chat.key, now):
                reason = self._dm_reason(dm_chat, now)
                if reason is not None:
                    content = self._draft(f"Send the partner a spontaneous DM. Reason: {reason}")
                    if content:
                        result = self._emit("dm", dm_chat, content, reason, now)
                        if result.get("ok"):
                            self._day.dm_sent += 1
                            self._day.last_dm_ts = now
                            self._day.last_chat_ts[dm_chat.key] = now
                        return result

            # 2) Ambient share: once a day, if she's in a good state.
            if dm_chat is not None and dm_on \
                    and not self._cap_reached(self._day.dm_sent, self.max_dm_per_day) \
                    and self._chat_allowed(dm_chat.key, now):
                hour = time.localtime(now).tm_hour
                if self._day.last_dm_ts == 0.0 and 10 <= hour < 20:
                    ambient = self.background.ambient_lines(
                        user_in_us=True, romantic=self.relationship.is_romantic()
                    )
                    if ambient and self.mood.current().values.get("energy", 50) > 55:
                        content = self._draft(
                            f"Share one small, specific thing from your day with the partner. "
                            f"Use this detail: {ambient[0]} Keep it under two sentences."
                        )
                        if content:
                            result = self._emit("dm", dm_chat, content, "ambient share", now)
                            if result.get("ok"):
                                self._day.dm_sent += 1
                                self._day.last_dm_ts = now
                                self._day.last_chat_ts[dm_chat.key] = now
                            return result

            # 3) A group post, on her own interests.
            if groups_on and self.group_chats \
                    and not self._cap_reached(self._day.group_sent, self.max_group_per_day):
                label = self.mood.current().label
                if label in {"playful", "happy", "excited", "proud", "calm"}:
                    for key in sorted(self.group_chats):
                        chat = self._parse_chat(key)
                        if chat is None or not self._chat_allowed(chat.key, now):
                            continue
                        topic = self._interest_for(now)
                        content = self._draft(
                            f"Write a short public group post about: {topic}. "
                            f"Rules: first person as {self.persona.name}; at most two sentences; "
                            "casual, dry, a little opinionated; NO mention of the partner, "
                            "the relationship, or being an AI; end with a hook people can reply to."
                        )
                        if content:
                            result = self._emit("group", chat, content, f"group post ({topic})", now)
                            if result.get("ok"):
                                self._day.group_sent += 1
                                self._day.last_chat_ts[chat.key] = now
                            return result
            self.stats["skipped"] += 1
            return {"decision": "nothing to send"}

    def _dm_reason(self, chat: ChatRef, now: float) -> str | None:
        row = self.context.db.query_one("SELECT last_active FROM chats WHERE id = ?", (chat.key,))
        last = float(row.get("last_active") or 0.0) if row else 0.0
        hours_idle = (now - last) / 3600.0 if last else 0.0
        values = self.mood.current().values
        if hours_idle >= self.silence_hours:
            if values.get("distance", 0) > 50 or values.get("insecurity", 0) > 50:
                return (
                    f"It's been {hours_idle:.0f} hours since you last talked and you've been "
                    "a bit quiet about it. Say something small that shows you've been thinking "
                    "about them — not a full confession, not an accusation. One or two lines."
                )
            if values.get("affection", 0) > 60:
                return (
                    f"It's been {hours_idle:.0f} hours of quiet and you're in a good, warm headspace. "
                    "A short 'thinking of you' note, specific to something small."
                )
        return None

    def _interest_for(self, now: float) -> str:
        interests = list(self.persona.interests) or ["hiking", "coffee", "music"]
        # Rotate deterministically by hour so it varies but is stable per hour.
        return interests[time.localtime(now).tm_hour % len(interests)]

    def _owner_dm_chat(self) -> ChatRef | None:
        if not self.owner_chats:
            return None
        for key in sorted(self.owner_chats):
            chat = self._parse_chat(key)
            if chat is not None and chat.kind == ChatKind.DM:
                return chat
        for key in sorted(self.owner_chats):
            chat = self._parse_chat(key)
            if chat is not None:
                return chat
        return None

    def _parse_chat(self, key: str) -> ChatRef | None:
        try:
            chat = ChatRef.parse(key)
        except ValueError:
            return None
        row = self.context.db.query_one("SELECT kind FROM chats WHERE id = ?", (chat.key,))
        if row is not None:
            chat = ChatRef(platform=chat.platform, chat_id=chat.chat_id, kind=row["kind"])
        return chat

    # ── drafting ────────────────────────────────────────────────────────────
    def _draft(self, instruction: str) -> str:
        label = self.mood.current().label
        system = (
            f"You are {self.persona.name}, in a real, unscripted moment. "
            f"Current state: you are {label}. {self.mood.describe()}\n"
            "Write exactly the message you would send, as you'd actually type it. "
            "No quotes, no labels, no emojis unless one truly fits. Output only the message."
        )
        try:
            response = self.context.router.chat(
                [Message.system(system), Message.user(instruction)],
                SamplingParams(temperature=0.9, max_tokens=160),
            )
        except Exception as exc:  # noqa: BLE001 - proactive failure = silence
            _log.warning("autonomy draft failed: %s", exc)
            return ""
        if not response.ok or not response.text.strip():
            return ""
        text = strip_robotic(response.text.strip())
        text = clamp_to_budget(text, (10, 300), soft=False)
        parts = split_messages(text, max_chars=300)
        return parts[0] if parts else ""

    # ── emitting with the approval flow ─────────────────────────────────────
    def _emit(self, kind: str, chat: ChatRef, content: str, reason: str, now: float) -> dict[str, Any]:
        proposal_id = ulid_now()
        self.stats["proposals"] += 1
        self._record(proposal_id, kind, chat, content, reason, "pending", now)
        if self.mode != "auto":
            _log.info("proposal %s [%s] %s: %s", proposal_id, kind, chat.key, content[:80])
            return {"ok": False, "proposal": proposal_id, "status": "pending",
                    "note": "held for approval (mode=suggest)"}
        result = self.gateway.send(chat.platform, chat, content)
        status = "sent" if result.ok else "failed"
        self._record(proposal_id, kind, chat, content, reason, status, now)
        if result.ok:
            self.stats["sent"] += 1
        return {"ok": result.ok, "proposal": proposal_id, "status": status,
                "error": result.error}

    def _record(self, proposal_id: str, kind: str, chat: ChatRef, content: str,
                reason: str, status: str, now: float) -> None:
        try:
            with self.context.db.transaction():
                self.context.db.execute(
                    """INSERT INTO proactive_log (id, kind, platform, chat_id, content, status, reason, decided_at, acted_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET status = excluded.status, acted_at = excluded.acted_at""",
                    (proposal_id, kind, chat.platform, chat.chat_id, content, status, reason, now,
                     now if status in {"sent", "failed", "approved", "denied"} else None),
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("proactive_log write failed: %s", exc)

    # ── approvals (CLI surface) ──────────────────────────────────────────────
    def approve(self, proposal_id: str) -> dict[str, Any]:
        row = self.context.db.query_one(
            "SELECT * FROM proactive_log WHERE id = ?", (proposal_id,)
        )
        if row is None:
            return {"ok": False, "error": f"no proposal {proposal_id!r}"}
        if row["status"] != "pending":
            return {"ok": False, "error": f"proposal is {row['status']}, not pending"}
        chat = ChatRef(platform=row["platform"], chat_id=row["chat_id"],
                       kind="group" if row["kind"] == "group" else "dm")
        result = self.gateway.send(chat.platform, chat, row["content"])
        status = "sent" if result.ok else "failed"
        try:
            self.context.db.execute(
                "UPDATE proactive_log SET status = ?, acted_at = ? WHERE id = ?",
                (status, time.time(), proposal_id),
            )
        except Exception:  # noqa: BLE001
            pass
        if result.ok:
            self.stats["sent"] += 1
        return {"ok": result.ok, "status": status, "error": result.error}

    def deny(self, proposal_id: str) -> dict[str, Any]:
        row = self.context.db.query_one("SELECT status FROM proactive_log WHERE id = ?", (proposal_id,))
        if row is None:
            return {"ok": False, "error": f"no proposal {proposal_id!r}"}
        if row["status"] != "pending":
            return {"ok": False, "error": f"proposal is {row['status']}, not pending"}
        self.context.db.execute(
            "UPDATE proactive_log SET status = 'denied', acted_at = ? WHERE id = ?",
            (time.time(), proposal_id),
        )
        return {"ok": True, "status": "denied"}

    def status(self) -> dict[str, Any]:
        pending = self.context.db.scalar(
            "SELECT COUNT(*) FROM proactive_log WHERE status = 'pending'", default=0
        )
        return {
            "mode": self.mode,
            "running": self._thread is not None and self._thread.is_alive(),
            "pending_proposals": int(pending),
            "today": {"dm": self._day.dm_sent, "group": self._day.group_sent},
            "stats": dict(self.stats),
        }
