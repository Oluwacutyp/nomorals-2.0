"""The autonomy agent: she initiates, on her own clock.

A real partner doesn't wait to be asked. This agent runs a heartbeat and,
from her *current mood*, the time of day, how long it's been quiet, and the
facts of her life (background pack), decides whether to:

* send the partner a spontaneous DM (check-in, miss-you, sharing her day), or
* post to one of the configured group chats, on a topic that fits her
  interests — never about the partner, never personal.

The decision is a strategy chain, not a hardcoded if-ladder: each
``Strategy`` (silence check-in, ambient share, group post) scores its
outreach from the live context; safety vetoes (quiet hours, daily caps,
per-chat intervals, feature flags) apply after selection; the best
eligible proposal above an adaptive threshold gets drafted and emitted.
Successes make her slightly bolder, failures and denials make her more
cautious — the threshold adapts, the safety rules never do.

Modes (``partner.autonomy_mode``):

* ``off``     — the agent does not start.
* ``suggest`` — every intended send becomes a *proposal* in ``proactive_log``
  (status=pending). Nothing leaves the machine until the owner approves it
  (``nm partner --approve <id>``). Opt-in for cautious setups.
* ``auto``    — sends go out directly, subject to quiet hours, per-day caps,
  per-chat minimum intervals, and the gateway's hourly window. Everything is
  still journaled in ``proactive_log`` and the audit trail.  A cap of ``0``
  means *unlimited* (the owner accepts the volume). This is the default.

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
from ..llm.brain import brain_for
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


# ── decision strategies ──────────────────────────────────────────────────
# The autonomy decision is a strategy chain, not a hardcoded if-ladder.
# Each strategy scores one kind of outreach from the live context; the
# agent picks the best eligible proposal above an adaptive threshold.
# Safety rules (quiet hours, caps, intervals, feature flags) are vetoes
# applied AFTER selection — they are never negotiable, but they are not
# the decision logic either.


@dataclass
class Proposal:
    """One scored outreach candidate from a strategy."""

    kind: str              # "dm" | "group"
    chat: ChatRef
    reason: str
    score: float           # 0..1 — higher means more worth sending now
    strategy: str
    draft_instruction: str


@dataclass
class DecisionContext:
    """Everything strategies may read.  Fresh per tick."""

    now: float
    mood_label: str
    mood_values: dict[str, float]
    dm_sent: int
    group_sent: int
    last_dm_ts: float
    last_chat_ts: dict[str, float]


class Strategy:
    """One outreach strategy.  Returns scored proposals (possibly none)."""

    name: str = "strategy"

    def evaluate(self, agent: "AutonomyAgent",
                 ctx: DecisionContext) -> list[Proposal]:
        raise NotImplementedError


class SilenceCheckinStrategy(Strategy):
    """A DM when the silence has run long — scored by how overdue it is
    and how much the current mood pulls toward contact."""

    name = "silence_checkin"

    def evaluate(self, agent: "AutonomyAgent",
                 ctx: DecisionContext) -> list[Proposal]:
        chat = agent._owner_dm_chat()
        if chat is None:
            return []
        row = agent.context.db.query_one(
            "SELECT last_active FROM chats WHERE id = ?", (chat.key,))
        last = float(row.get("last_active") or 0.0) if row else 0.0
        hours_idle = (ctx.now - last) / 3600.0 if last else 0.0
        if hours_idle < agent.silence_hours:
            return []
        reason = agent._dm_reason(chat, ctx.now)
        if reason is None:
            return []
        overdue = min(1.0, (hours_idle - agent.silence_hours + 2.0) / 12.0)
        mood_pull = max(ctx.mood_values.get("distance", 0.0),
                        ctx.mood_values.get("insecurity", 0.0),
                        ctx.mood_values.get("affection", 0.0)) / 100.0
        score = 0.40 + 0.40 * overdue + 0.20 * mood_pull
        return [Proposal(
            kind="dm", chat=chat, reason=reason,
            score=min(1.0, score), strategy=self.name,
            draft_instruction=(
                f"Send the partner a spontaneous DM. Reason: {reason}"),
        )]


class AmbientShareStrategy(Strategy):
    """Once a day, share one small real thing from her day — only when
    her energy is up and she hasn't already DM'd today."""

    name = "ambient_share"

    def evaluate(self, agent: "AutonomyAgent",
                 ctx: DecisionContext) -> list[Proposal]:
        chat = agent._owner_dm_chat()
        if chat is None or ctx.last_dm_ts != 0.0:
            return []
        hour = time.localtime(ctx.now).tm_hour
        if not 10 <= hour < 20:
            return []
        energy = ctx.mood_values.get("energy", 50.0)
        if energy <= 55:
            return []
        ambient = agent.background.ambient_lines(
            user_in_us=True, romantic=agent.relationship.is_romantic())
        if not ambient:
            return []
        score = 0.45 + 0.30 * ((energy - 55.0) / 45.0)
        return [Proposal(
            kind="dm", chat=chat, reason="ambient share",
            score=min(1.0, score), strategy=self.name,
            draft_instruction=(
                "Share one small, specific thing from your day with the "
                f"partner. Use this detail: {ambient[0]} "
                "Keep it under two sentences."),
        )]


class GroupPostStrategy(Strategy):
    """A public group post on her own interests when the mood is social.
    One proposal per configured group chat — the per-chat interval veto
    picks which one is actually eligible."""

    name = "group_post"

    def evaluate(self, agent: "AutonomyAgent",
                 ctx: DecisionContext) -> list[Proposal]:
        if not agent.group_chats:
            return []
        if ctx.mood_label not in agent.group_moods:
            return []
        boost = 0.10 if ctx.mood_label in {"playful", "excited"} else 0.0
        proposals = []
        for key in sorted(agent.group_chats):
            chat = agent._parse_chat(key)
            if chat is None:
                continue
            topic = agent._interest_for(ctx.now)
            proposals.append(Proposal(
                kind="group", chat=chat,
                reason=f"group post ({topic})",
                score=min(1.0, 0.50 + boost), strategy=self.name,
                draft_instruction=(
                    f"Write a short public group post about: {topic}. "
                    f"Rules: first person as {agent.persona.name}; at most two "
                    "sentences; casual, dry, a little opinionated; NO mention "
                    "of the partner, the relationship, or being an AI; end "
                    "with a hook people can reply to."),
            ))
        return proposals


class AutonomyAgent:
    """Heartbeat-driven proactive messaging, with an approval flow."""

    def __init__(
        self,
        context: Any,
        brain: Any,
        gateway: ChatGateway,
        *,
        mode: str = "auto",
        owner_chats: set[str] | None = None,
        group_chats: set[str] | None = None,
        quiet_start: int = 22,
        quiet_end: int = 8,
        max_dm_per_day: int = 6,
        max_group_per_day: int = 2,
        heartbeat_seconds: float = 900.0,
        silence_hours_for_checkin: float = 6.0,
        group_moods: set[str] | None = None,
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
        #: moods in which a group post is socially appropriate — data, not
        #: a hardcoded set buried in the decision logic.
        self.group_moods = set(group_moods or
                               {"playful", "happy", "excited", "proud", "calm"})
        #: the strategy chain, evaluated in order every tick
        self._strategies: list[Strategy] = [
            SilenceCheckinStrategy(),
            AmbientShareStrategy(),
            GroupPostStrategy(),
        ]
        #: adaptive send threshold: successes make her slightly bolder,
        #: failures and denials make her more cautious.  Bounded.
        self._threshold = 0.5
        #: per-strategy win/loss counters — the learning memory.  "wins"
        #: counts proposals that led somewhere (sent + owner reply);
        #: "proposed"/"sent"/"denied" are raw counters.
        self.strategy_stats: dict[str, dict[str, int]] = {
            s.name: {"proposed": 0, "sent": 0, "wins": 0, "denied": 0}
            for s in self._strategies
        }
        #: proposal id → strategy name, for outcome feedback.
        self._strategy_by_proposal: dict[str, str] = {}

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
        """One heartbeat: strategies propose, safety vetoes, best eligible
        proposal above the adaptive threshold gets drafted and emitted."""
        from .features import feature_enabled

        now = now if now is not None else time.time()
        with self._lock:
            self.stats["ticks"] += 1
            if self._in_quiet_hours(now):
                self.stats["skipped"] += 1
                return {"decision": "quiet hours"}
            self._roll_day(now)
            ctx = DecisionContext(
                now=now,
                mood_label=self.mood.current().label,
                mood_values=dict(self.mood.current().values),
                dm_sent=self._day.dm_sent,
                group_sent=self._day.group_sent,
                last_dm_ts=self._day.last_dm_ts,
                last_chat_ts=dict(self._day.last_chat_ts),
            )
            proposals: list[Proposal] = []
            for strategy in self._strategies:
                try:
                    new = strategy.evaluate(self, ctx) or []
                    if new:
                        # Custom/plugged strategies are welcome — their
                        # counters are created on first sight.
                        self.strategy_stats.setdefault(
                            strategy.name,
                            {"proposed": 0, "sent": 0, "wins": 0,
                             "denied": 0})["proposed"] += len(new)
                    proposals.extend(new)
                except Exception:  # noqa: BLE001 - one bad strategy never kills the tick
                    _log.exception("autonomy strategy %s failed", strategy.name)
            eligible = [p for p in proposals
                        if self._policy_allows(p, ctx, now, feature_enabled)]
            if not eligible:
                self.stats["skipped"] += 1
                return {"decision": "nothing to send",
                        "proposals": len(proposals),
                        "strategies": sorted({p.strategy for p in proposals})}
            best = max(eligible, key=lambda p: p.score)
            if best.score < self._threshold:
                self.stats["skipped"] += 1
                return {"decision": "below threshold",
                        "best": round(best.score, 3),
                        "threshold": round(self._threshold, 3),
                        "strategy": best.strategy}
            content = self._draft(best.draft_instruction)
            if not content:
                self.stats["skipped"] += 1
                return {"decision": "draft failed",
                        "strategy": best.strategy}
            result = self._emit(best.kind, best.chat, content, best.reason,
                                now, strategy=best.strategy)
            if result.get("ok"):
                if best.kind == "dm":
                    self._day.dm_sent += 1
                    self._day.last_dm_ts = now
                else:
                    self._day.group_sent += 1
                self._day.last_chat_ts[best.chat.key] = now
                self.strategy_stats.setdefault(
                    best.strategy,
                    {"proposed": 0, "sent": 0, "wins": 0,
                     "denied": 0})["sent"] += 1
                self._adapt_threshold(success=True)
            else:
                # held for approval (suggest mode) is not a failure — only
                # a genuinely failed send tightens the threshold
                if result.get("status") == "failed":
                    self._adapt_threshold(success=False)
            result["strategy"] = best.strategy
            result["score"] = round(best.score, 3)
            return result

    def _policy_allows(self, proposal: Proposal, ctx: DecisionContext,
                       now: float, feature_enabled: Any) -> bool:
        """Safety vetoes.  Never negotiable, evaluated after the
        strategies score — caps, intervals, feature flags, chat kind."""
        if proposal.kind == "dm":
            if not feature_enabled(self.context, "proactive_dm"):
                return False
            if self._cap_reached(ctx.dm_sent, self.max_dm_per_day):
                return False
        else:
            if not feature_enabled(self.context, "group_posts"):
                return False
            if self._cap_reached(ctx.group_sent, self.max_group_per_day):
                return False
        return self._chat_allowed(proposal.chat.key, now)

    def _adapt_threshold(self, *, success: bool) -> None:
        """Adaptive eagerness: successes make her slightly bolder,
        failures and denials make her more cautious.  Bounded
        [0.3, 0.9] — the threshold adapts, the safety rules don't."""
        if success:
            self._threshold = max(0.3, self._threshold * 0.98)
        else:
            self._threshold = min(0.9, self._threshold + 0.05)

    # ── outcome feedback ─────────────────────────────────────────────────────
    def note_outcome(self, proposal_id: str,
                     outcome: str) -> dict[str, Any]:
        """Record what happened *after* a proactive send.

        ``outcome``: ``"replied"`` (owner answered the DM — the strongest
        positive signal), ``"ignored"`` (sent, no reply within a day), or
        ``"denied"`` (owner rejected a pending proposal). Learning from
        outcomes, not just sends, is what makes the threshold honest:
        a sent message nobody answers should not make her bolder.
        """
        outcome = (outcome or "").strip().lower()
        if outcome not in {"replied", "ignored", "denied"}:
            return {"ok": False, "error": "outcome must be "
                    "replied|ignored|denied"}
        with self._lock:
            strategy = self._strategy_by_proposal.get(proposal_id, "")
            stats = self.strategy_stats.get(strategy) if strategy else None
            if outcome == "replied":
                # A reply means the outreach landed: bolder, and the
                # strategy that produced it gets the win.
                self._threshold = max(0.3, self._threshold * 0.95)
                if stats is not None:
                    stats["wins"] += 1
                learned = "owner replied to proactive outreach"
            elif outcome == "denied":
                self._threshold = min(0.9, self._threshold + 0.05)
                if stats is not None:
                    stats["denied"] += 1
                learned = "owner denied the proposal"
            else:  # ignored
                # Silence after a send: slightly more cautious, not a
                # punishment — half the failure step.
                self._threshold = min(0.9, self._threshold + 0.025)
                learned = "proactive send went unanswered"
            self._ledger("outcome", proposal_id,
                         f"outcome={outcome} for proposal {proposal_id}",
                         learned=learned,
                         metadata={"outcome": outcome,
                                   "strategy": strategy,
                                   "threshold": round(self._threshold, 3)})
            return {"ok": True, "outcome": outcome,
                    "strategy": strategy,
                    "threshold": round(self._threshold, 3)}

    def on_owner_reply(self, chat_key: str) -> dict[str, Any]:
        """The owner replied in a chat — credit the most recent proactive
        DM proposal there as ``replied``.

        The runtime's inbound message path should call this (cheap, never
        raises) so the agent learns which outreach actually lands.
        """
        try:
            row = self.context.db.query_one(
                """SELECT id FROM proactive_log
                   WHERE kind = 'dm' AND status = 'sent'
                     AND (platform || ':' || chat_id) = ?
                   ORDER BY acted_at DESC LIMIT 1""",
                (chat_key,),
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("on_owner_reply lookup failed: %s", exc)
            return {"ok": False, "error": "lookup failed"}
        if row is None:
            return {"ok": True, "noted": False,
                    "note": "no proactive DM to credit in this chat"}
        return {**self.note_outcome(row["id"], "replied"), "noted": True}

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
            response = brain_for(self.context).chat(
                [Message.system(system), Message.user(instruction)],
                SamplingParams(temperature=0.9, max_tokens=160),
            task_kind="plan")
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
    def _emit(self, kind: str, chat: ChatRef, content: str, reason: str,
              now: float, strategy: str = "") -> dict[str, Any]:
        proposal_id = ulid_now()
        self.stats["proposals"] += 1
        self._record(proposal_id, kind, chat, content, reason, "pending", now)
        self._strategy_by_proposal[proposal_id] = strategy
        self._ledger("proposal", proposal_id,
                     f"[{kind}] {reason}: {content[:120]}",
                     metadata={"chat": chat.key, "reason": reason,
                               "strategy": strategy})
        if self.mode != "auto":
            _log.info("proposal %s [%s] %s: %s", proposal_id, kind, chat.key, content[:80])
            return {"ok": False, "proposal": proposal_id, "status": "pending",
                    "note": "held for approval (mode=suggest)"}
        result = self.gateway.send(chat.platform, chat, content)
        status = "sent" if result.ok else "failed"
        self._record(proposal_id, kind, chat, content, reason, status, now)
        self._ledger("send" if result.ok else "send_failed", proposal_id,
                     f"[{kind}] {reason}: {content[:120]}",
                     ok=result.ok,
                     learned="" if result.ok else str(result.error or "")[:200],
                     metadata={"chat": chat.key})
        try:
            from ..core.events import Event, global_bus

            global_bus.publish(Event(
                topic=f"autonomy.{status}",
                data={"proposal_id": proposal_id, "kind": kind,
                      "chat": chat.key, "reason": reason},
                source="nomorals.agents.autonomy"))
        except Exception:  # noqa: BLE001 - telemetry is fail-open
            _log.debug("autonomy bus publish failed", exc_info=True)
        if result.ok:
            self.stats["sent"] += 1
        return {"ok": result.ok, "proposal": proposal_id, "status": status,
                "error": result.error}

    def _ledger(self, kind: str, ref_id: str, summary: str, *,
                ok: bool = True, learned: str = "",
                metadata: dict[str, Any] | None = None) -> None:
        """Journal to the unified autonomy ledger.  Never raises."""
        try:
            from .autonomy_ledger import record_ledger

            record_ledger(self.context, "autonomy", kind, ref_id, summary,
                          ok=ok, learned=learned, metadata=metadata)
        except Exception:  # noqa: BLE001
            _log.debug("autonomy ledger write failed", exc_info=True)

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
        # a denial is feedback: she gets more cautious about proposing
        self._threshold = min(0.9, self._threshold + 0.05)
        strategy = self._strategy_by_proposal.get(proposal_id, "")
        if strategy in self.strategy_stats:
            self.strategy_stats[strategy]["denied"] += 1
        return {"ok": True, "status": "denied"}

    def status(self) -> dict[str, Any]:
        try:
            pending = self.context.db.scalar(
                "SELECT COUNT(*) FROM proactive_log WHERE status = 'pending'",
                default=0,
            )
        except Exception:  # noqa: BLE001 - status must never raise
            pending = 0
        return {
            "mode": self.mode,
            "running": self._thread is not None and self._thread.is_alive(),
            "pending_proposals": int(pending),
            "today": {"dm": self._day.dm_sent, "group": self._day.group_sent},
            "stats": dict(self.stats),
            "threshold": round(self._threshold, 3),
            "strategies": [s.name for s in self._strategies],
            "strategy_stats": {k: dict(v)
                               for k, v in self.strategy_stats.items()},
            "group_moods": sorted(self.group_moods),
        }
