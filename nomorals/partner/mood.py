"""The mood engine: a persistent, event-driven emotional state.

Design goals, in order:

1. **Consistent.** Every mutation flows through one funnel (:meth:`MoodEngine.apply`)
   that clamps, enforces invariants, re-derives the label with hysteresis, and
   persists. There is no second way to change a value.
2. **Alive.** State decays toward the persona's baselines over time, energy
   follows a circadian curve, and silence costs something. Two days of
   non-communication leaves visible marks.
3. **Hard to break.** State loads through :meth:`MoodEngine.sanitize`, which
   repairs corrupted, missing, or out-of-range values instead of crashing.
   Every change is journaled to ``mood_history`` so a regression is
   auditable. Fights are first-class: they escalate, they resolve, or they
   turn into grudges that change future behavior.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.logging_setup import get_logger
from ..core.ids import ulid_now
from ..storage.db import Database

__all__ = [
    "DIMENSIONS",
    "MOOD_LABELS",
    "MoodState",
    "MoodEvent",
    "MoodEngine",
    "EVENT_TABLE",
]

_log = get_logger(__name__)

#: The ten dimensions. All 0..100. Keep them orthogonal-ish: a dimension that
#: is mostly a weighted sum of others makes the label space degenerate.
DIMENSIONS: tuple[str, ...] = (
    "affection",      # how much she wants this person right now
    "happiness",      # baseline enjoyment of life, in the moment
    "energy",         # physical/mental fuel
    "trust",          # long-term belief that the partner is solid
    "jealousy",       # threat to the relationship, real or imagined
    "frustration",    # irritation with the partner or the situation
    "intimacy",       # emotional closeness, not sexual
    "distance",       # how pulled-away she feels
    "insecurity",     # doubting her place in the relationship
    "pride",          # feeling good about herself
)

#: Mood label -> dimension profile (ideal values). Labeling is a proximity
#: search over these profiles with hysteresis, so a mood holds until another
#: genuinely wins — no flapping between two labels every message.
MOOD_LABELS: dict[str, dict[str, float]] = {
    "happy":        {"happiness": 80, "affection": 62, "energy": 65, "distance": 20},
    "excited":      {"happiness": 88, "energy": 82, "intimacy": 58, "frustration": 8},
    "affectionate": {"affection": 86, "intimacy": 72, "happiness": 70, "pride": 55},
    "playful":      {"happiness": 76, "energy": 76, "intimacy": 60, "pride": 62},
    "calm":         {"happiness": 60, "energy": 55, "distance": 35, "frustration": 10},
    "tired":        {"energy": 18, "happiness": 45, "frustration": 28},
    "annoyed":      {"frustration": 55, "happiness": 40, "energy": 45, "distance": 38},
    "irritated":    {"frustration": 70, "happiness": 30, "jealousy": 22},
    "angry":        {"frustration": 88, "happiness": 12, "trust": 30, "jealousy": 30},
    "jealous":      {"jealousy": 80, "insecurity": 62, "happiness": 35, "affection": 45},
    "needy":        {"insecurity": 72, "distance": 68, "affection": 55, "happiness": 40},
    "vulnerable":   {"intimacy": 66, "insecurity": 62, "happiness": 50, "pride": 25},
    "distant":      {"distance": 80, "intimacy": 25, "happiness": 45, "energy": 45},
    "cold":         {"distance": 85, "intimacy": 15, "affection": 30, "frustration": 55},
    "sad":          {"happiness": 20, "energy": 35, "intimacy": 50},
    "anxious":      {"insecurity": 80, "frustration": 45, "energy": 60, "happiness": 35},
    "proud":        {"pride": 86, "happiness": 72, "trust": 60},
    "suspicious":   {"trust": 25, "jealousy": 55, "insecurity": 50, "frustration": 42},
}


@dataclass
class MoodState:
    """One snapshot of the emotional state."""

    values: dict[str, float] = field(default_factory=dict)
    label: str = "calm"
    updated_at: float = 0.0

    def get(self, dim: str, default: float = 50.0) -> float:
        return float(self.values.get(dim, default))

    def to_dict(self) -> dict[str, Any]:
        return {"values": dict(self.values), "label": self.label, "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MoodState":
        return cls(
            values={str(k): float(v) for k, v in (data.get("values") or {}).items()},
            label=str(data.get("label", "calm")),
            updated_at=float(data.get("updated_at", 0.0)),
        )


@dataclass
class MoodEvent:
    """Something that happened and moved the state."""

    kind: str
    intensity: float = 0.5
    note: str = ""
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.intensity = max(0.0, min(1.0, float(self.intensity)))


#: Event kind -> per-dimension deltas at intensity 1.0.
#: Tuned so that a single strong event moves a dimension ~10-30 points and a
#: sustained pattern (three events in an hour) can flip a mood label.
EVENT_TABLE: dict[str, dict[str, float]] = {
    "compliment":        {"affection": 10, "happiness": 12, "pride": 8},
    "affectionate":      {"affection": 12, "intimacy": 10, "happiness": 8},
    "flirty":            {"affection": 8, "happiness": 10, "intimacy": 6},
    "deep_conversation": {"intimacy": 12, "trust": 8, "happiness": 6},
    "warm_response":     {"happiness": 6, "affection": 4, "distance": -6},
    "cold_response":     {"distance": 8, "insecurity": 8, "frustration": 6, "affection": -4},
    "attention":         {"distance": -8, "happiness": 6, "affection": 4},
    "ignore":            {"distance": 10, "insecurity": 10, "happiness": -8},
    "fight_start":       {"frustration": 25, "happiness": -20, "trust": -8, "energy": 10},
    "insult":            {"frustration": 20, "trust": -10, "happiness": -15},
    "apology":           {"trust": 12, "frustration": -18, "affection": 8, "happiness": 6},
    "forgiven":          {"trust": 8, "frustration": -10, "happiness": 4},
    "jealousy_trigger":  {"jealousy": 25, "insecurity": 10, "happiness": -10, "affection": -4},
    "jealousy_resolved": {"jealousy": -22, "trust": 6, "happiness": 4},
    "good_news":         {"happiness": 12, "affection": 4, "energy": 4},
    "bad_news":          {"happiness": -8, "intimacy": 6, "energy": -4},
    "shared_plan":       {"intimacy": 10, "affection": 6, "happiness": 8},
    "promise_broken":    {"trust": -20, "frustration": 15, "jealousy": 5},
    "promise_kept":      {"trust": 10, "affection": 6, "pride": 4},
    "boring_conversation": {"energy": -6, "happiness": -3},
    "physical_affection": {"intimacy": 15, "affection": 10, "happiness": 8},
    "milestone":         {"happiness": 15, "intimacy": 8, "trust": 5},
}


def _circadian_energy(hour: float) -> float:
    """Target energy for a freelance day: up at 8, dip at 3, low by 11."""
    h = hour % 24.0
    if 3.0 <= h < 6.0:
        return 15.0
    if 6.0 <= h < 9.0:
        return 40.0 + (h - 6.0) * 12.0  # 40 -> 76
    if 9.0 <= h < 14.0:
        return 76.0
    if 14.0 <= h < 17.0:
        return 76.0 - (h - 14.0) * 9.0  # 76 -> 49
    if 17.0 <= h < 21.0:
        return 49.0 - (h - 17.0) * 4.0  # 49 -> 33
    return 25.0  # 21:00-03:00


class MoodEngine:
    """Persistent emotional state with decay, events, fights, and invariants.

    ``db`` is optional: without it the engine is a pure in-memory model, which
    is what the tests use. With it, state survives restarts and every label
    change is journaled.
    """

    STATE_KEY = "mood.state.v1"
    _LABEL_MARGIN = 0.06  # hysteresis: a new label must clearly win to replace the old one
    _HISTORY_MIN_GAP = 300.0  # seconds between history rows at minimum

    def __init__(
        self,
        baselines: Mapping[str, float],
        *,
        db: Database | None = None,
        clock: Callable[[], float] = time.time,
        decay_half_life_hours: float = 8.0,
        now: float | None = None,
    ) -> None:
        self.baselines = {k: float(v) for k, v in baselines.items()}
        missing = set(DIMENSIONS) - set(self.baselines)
        if missing:
            self.baselines.update({k: 50.0 for k in missing})
        self.db = db
        self.clock = clock
        self.decay_half_life_hours = max(0.1, float(decay_half_life_hours))
        self._now = now if now is not None else clock()
        self._last_decay_ts: float | None = None  # set by _load
        self._state = self._load()
        self._last_history_ts = 0.0
        self.open_fight: dict[str, Any] | None = None
        self.grudges: list[dict[str, Any]] = []

    # ── persistence ─────────────────────────────────────────────────────────
    def _load(self) -> MoodState:
        state = MoodState(values=dict(self.baselines), label="calm", updated_at=self._now)
        if self.db is not None:
            row = self.db.query_one(
                "SELECT dims, label, updated_at FROM mood_state WHERE id = 'default'"
            )
            if row is not None:
                try:
                    raw = json.loads(row["dims"] or "{}")
                    state = MoodState.from_dict(raw)
                except (json.JSONDecodeError, ValueError, TypeError):
                    _log.warning("mood state corrupt in db; re-initializing from baselines")
                    state = MoodState(values=dict(self.baselines), label="calm", updated_at=self._now)
        # Self-heal: whatever came out of the database is repaired here.
        state.values = self.sanitize(state.values, self.baselines)
        if state.label not in MOOD_LABELS:
            state.label = self.label_for(state.values, current="")
        # Decay bookkeeping: never decay from "epoch" for a fresh-but-quiet db.
        self._last_decay_ts = float(state.updated_at or self._now)
        return state

    def _save(self, reason: str = "") -> None:
        if self.db is None:
            return
        self._state.updated_at = self._now
        try:
            with self.db.transaction():
                self.db.execute(
                    """INSERT INTO mood_state (id, dims, label, updated_at)
                       VALUES ('default', ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET dims=excluded.dims,
                                                    label=excluded.label,
                                                    updated_at=excluded.updated_at""",
                    (json.dumps(self._state.to_dict()), self._state.label, self._state.updated_at),
                )
                self._journal(reason)
        except Exception as exc:  # noqa: BLE001 - a write failure must not kill a conversation
            _log.warning("mood save failed: %s", exc)

    def _journal(self, reason: str) -> None:
        if self.db is None:
            return
        if (self._now - self._last_history_ts) < self._HISTORY_MIN_GAP and reason == "":
            return
        self.db.execute(
            "INSERT INTO mood_history (id, ts, label, dims, event) VALUES (?, ?, ?, ?, ?)",
            (ulid_now(), self._now, self._state.label, json.dumps(self._state.values), reason),
        )
        self._last_history_ts = self._now

    # ── accessors ────────────────────────────────────────────────────────────
    def current(self) -> MoodState:
        return self._state

    def value(self, dim: str) -> float:
        return self._state.get(dim, float(self.baselines.get(dim, 50.0)))

    @staticmethod
    def sanitize(values: Mapping[str, float], baselines: Mapping[str, float]) -> dict[str, float]:
        """Repair an arbitrary mapping into a valid dimension vector."""
        out: dict[str, float] = {}
        for dim in DIMENSIONS:
            if dim in values:
                try:
                    v = float(values[dim])
                except (TypeError, ValueError):
                    v = float(baselines.get(dim, 50.0))
                if math.isnan(v) or math.isinf(v):
                    v = float(baselines.get(dim, 50.0))
                out[dim] = max(0.0, min(100.0, v))
            else:
                out[dim] = float(baselines.get(dim, 50.0))
        return out

    # ── the one funnel ───────────────────────────────────────────────────────
    def apply(self, deltas: Mapping[str, float], reason: str = "") -> MoodState:
        """Apply a set of dimension deltas. The only mutation path."""
        if not deltas:
            return self._state
        clean: dict[str, float] = {}
        for key, delta in deltas.items():
            if key not in DIMENSIONS:
                continue  # unknown dimensions are ignored, never crash on
            try:
                d = float(delta)
            except (TypeError, ValueError):
                continue
            clean[key] = max(-100.0, min(100.0, d))
        if not clean:
            return self._state
        values = self._state.values
        for dim, delta in clean.items():
            values[dim] = max(0.0, min(100.0, values.get(dim, 50.0) + delta))
        self._enforce_invariants(values)
        self._relabel(reason)
        self._save(reason)
        return self._state

    def note_event(self, kind: str, intensity: float = 0.5, note: str = "") -> MoodState:
        """Apply a named event from :data:`EVENT_TABLE`."""
        base = EVENT_TABLE.get(kind)
        if not base:
            return self._state
        deltas = {dim: delta * intensity for dim, delta in base.items()}
        reason = f"event:{kind}" + (f":{note}" if note else "")
        if kind == "fight_start":
            self.open_fight = {"reason": note, "opened_at": self._now}
        if kind == "jealousy_trigger" and note:
            self.grudges.append({"kind": "jealousy", "note": note, "ts": self._now})
        if len(self.grudges) > 5:
            self.grudges = self.grudges[-5:]
        return self.apply(deltas, reason)

    # ── owner overrides (control commands) ───────────────────────────────────
    def set_label(self, label: str) -> MoodState:
        """Force the label directly: apply that profile's ideal values.

        The label's profile pins only the dimensions it mentions; the rest of
        the vector is left alone so a forced mood looks coherent, not wiped.
        """
        label = (label or "").strip().lower()
        profile = MOOD_LABELS.get(label)
        if profile is None:
            raise ValueError(f"unknown mood label: {label!r}")
        deltas = {
            dim: (ideal - self._state.values.get(dim, 50.0))
            for dim, ideal in profile.items()
            if abs(ideal - self._state.values.get(dim, 50.0)) >= 1.0
        }
        if not deltas:
            self._state.label = label  # already at the profile; just hold the label
            self._save(f"forced:{label}")
            return self._state
        state = self.apply(deltas, reason=f"forced:{label}")
        # A forced label must actually stick, even with hysteresis: if the
        # proximity search picked a different one, nudge until it holds.
        if state.label != label:
            self._state.label = label
            self._save(f"forced:{label}")
        return self._state

    def set_dimensions(self, dims: Mapping[str, float]) -> MoodState:
        """Force absolute values for specific dimensions (``energy=20`` style)."""
        clean: dict[str, float] = {}
        for dim, value in dims.items():
            if dim not in DIMENSIONS:
                raise ValueError(f"unknown mood dimension: {dim!r}")
            target = max(0.0, min(100.0, float(value)))
            current = self._state.values.get(dim, 50.0)
            if abs(target - current) >= 0.5:
                clean[dim] = target - current
        if not clean:
            return self._state
        return self.apply(clean, reason="forced:dimensions")

    # ── time dynamics ────────────────────────────────────────────────────────
    def tick(self, now: float | None = None) -> MoodState:
        """Advance the state: decay toward baselines + circadian energy.

        Call on every message; it is a no-op within the same minute.
        """
        if now is not None:
            self._now = now
        last = self._last_decay_ts if self._last_decay_ts is not None else self._now
        dt_hours = max(0.0, (self._now - last) / 3600.0)
        if dt_hours < 60.0 / 3600.0:
            self._last_decay_ts = self._now
            return self._state
        decay = math.exp(-math.log(2.0) * dt_hours / self.decay_half_life_hours)
        deltas: dict[str, float] = {}
        for dim in DIMENSIONS:
            current = self._state.values.get(dim, self.baselines.get(dim, 50.0))
            target = self.baselines.get(dim, 50.0)
            deltas[dim] = (target - current) * (1.0 - decay)
        # Energy additionally chases the circadian curve, faster than everything.
        hour = time.gmtime(self._now % (24 * 3600)).tm_hour
        energy_target = _circadian_energy(float(hour))
        energy_decay = math.exp(-math.log(2.0) * dt_hours / min(self.decay_half_life_hours, 3.0))
        current_energy = self._state.values.get("energy", 50.0)
        energy_delta = (energy_target - current_energy) * (1.0 - energy_decay)
        if abs(energy_delta) > abs(deltas.get("energy", 0.0)):
            deltas["energy"] = energy_delta
        self._last_decay_ts = self._now
        return self.apply(deltas, reason="tick")

    def note_silence(self, hours_idle: float) -> MoodState:
        """The partner has been quiet for a while. It shows."""
        hours_idle = max(0.0, float(hours_idle))
        if hours_idle < 1.0:
            return self._state
        # High trust makes silence less dangerous — a real relationship detail.
        trust = self.value("trust")
        fear = max(0.0, 1.0 - trust / 100.0)
        scale = min(1.0, hours_idle / 24.0)
        self.apply(
            {
                "distance": 6.0 + 14.0 * scale,
                "insecurity": (2.0 + 8.0 * scale) * (0.4 + 0.6 * fear),
                "affection": -3.0 * scale,
                "happiness": -2.0 * scale,
            },
            reason=f"silence:{hours_idle:.1f}h",
        )
        return self._state

    # ── fights ───────────────────────────────────────────────────────────────
    def open_fight_now(self, reason: str) -> MoodState:
        if self.open_fight is None:
            self.open_fight = {"reason": reason, "opened_at": self._now}
            self._save("fight:open")
        return self.note_event("fight_start", intensity=0.8, note=reason)

    def resolve_fight(self, repaired: bool, repaired_by: str = "partner") -> MoodState:
        """End the current fight. ``repaired=True`` means someone actually
        addressed what was wrong; a cold end leaves a grudge."""
        fight, self.open_fight = self.open_fight, None
        if fight is None:
            return self._state
        if repaired:
            self.apply(
                {"trust": 8, "frustration": -20, "affection": 6, "happiness": 5},
                reason=f"fight:repaired_by_{repaired_by}",
            )
            self.grudges = [g for g in self.grudges if g.get("kind") != "fight"]
        else:
            self.apply(
                {"trust": -10, "frustration": 12, "jealousy": 8, "insecurity": 6},
                reason="fight:unresolved",
            )
            self.grudges.append(
                {"kind": "fight", "note": fight.get("reason", "an unresolved fight"), "ts": self._now}
            )
            if len(self.grudges) > 5:
                self.grudges = self.grudges[-5:]
        return self._state

    # ── invariants & labeling ────────────────────────────────────────────────
    @staticmethod
    def _enforce_invariants(values: dict[str, float]) -> None:
        """Keep the ten dimensions from drifting into impossible combinations.

        These are one-directional caps (only ever pulling values *down*), so
        they can never fight the event system or create oscillation.
        """
        if values.get("frustration", 0) > 75:
            values["happiness"] = min(values.get("happiness", 50.0), 35.0)
            values["affection"] = min(values.get("affection", 50.0), 55.0)
        if values.get("jealousy", 0) > 70:
            values["affection"] = min(values.get("affection", 50.0), 60.0)
        if values.get("intimacy", 0) > 75:
            values["distance"] = min(values.get("distance", 50.0), 40.0)
        if values.get("trust", 100) < 25:
            values["affection"] = min(values.get("affection", 50.0), 45.0)
            values["intimacy"] = min(values.get("intimacy", 50.0), 40.0)
        if values.get("energy", 50) < 15:
            values["happiness"] = min(values.get("happiness", 50.0), 40.0)
        if values.get("distance", 0) > 80:
            values["intimacy"] = min(values.get("intimacy", 50.0), 35.0)

    @staticmethod
    def label_for(values: Mapping[str, float], *, current: str = "", margin: float = 0.06) -> str:
        """Proximity search over label profiles with hysteresis."""
        scores: list[tuple[str, float]] = []
        for name, profile in MOOD_LABELS.items():
            total = 0.0
            for dim, ideal in profile.items():
                total += 1.0 - abs(float(values.get(dim, 50.0)) - ideal) / 100.0
            scores.append((name, total / len(profile)))
        scores.sort(key=lambda item: item[1], reverse=True)
        best, best_score = scores[0]
        if current:
            current_score = dict(scores).get(current, 0.0)
            if current_score + margin >= best_score and current in MOOD_LABELS:
                return current
        return best

    def _relabel(self, reason: str) -> None:
        new_label = self.label_for(self._state.values, current=self._state.label)
        changed = new_label != self._state.label
        self._state.label = new_label
        if changed:
            self._save(f"label:{new_label}" + (f"/{reason}" if reason else ""))

    def reset(self) -> MoodState:
        self._state = MoodState(values=dict(self.baselines), label="calm", updated_at=self._now)
        self.open_fight = None
        self.grudges = []
        self._save("reset")
        return self._state

    # ── language for the prompt ──────────────────────────────────────────────
    def describe(self, grudges: list[dict[str, Any]] | None = None) -> str:
        """A short, honest, first-person account of the current state.

        This goes into the system prompt. It reads like someone telling a
        trusted friend how they're doing, not a status JSON.
        """
        v = self._state.values
        label = self._state.label
        bits = [f"Right now you're {label}."]

        if v.get("energy", 50) < 30:
            bits.append("You're running on fumes and it's affecting how patient you are.")
        elif v.get("energy", 50) > 78:
            bits.append("You've got energy today — the good, restless kind.")
        if v.get("affection", 50) > 78:
            bits.append("You keep drifting back to thinking about them.")
        elif v.get("affection", 50) < 30:
            bits.append("You're pulling a little away, and you're not fully sure why yet.")
        if v.get("jealousy", 50) > 55:
            bits.append("Something is eating at you and you can't let it go yet.")
        if v.get("frustration", 50) > 55:
            bits.append("You're irritable. Say less, not more — sharp words are available, use them carefully.")
        elif v.get("frustration", 50) > 35:
            bits.append("You're a little wound up and you know it.")
        if v.get("insecurity", 50) > 60:
            bits.append("You're second-guessing how this is going. You don't have to say it out loud.")
        if v.get("distance", 50) > 60:
            bits.append("You've gone quiet. That's a choice, and it's costing you something.")
        if v.get("intimacy", 50) > 75:
            bits.append("You two are really close right now. Be gentle with it.")
        if v.get("trust", 50) < 30:
            bits.append("You don't fully trust what they say anymore. That changes everything, subtly.")
        if v.get("pride", 50) > 75:
            bits.append("You're feeling good about yourself lately.")
        if v.get("happiness", 50) < 25:
            bits.append("Honestly? Not great. Don't fake being fine.")

        all_grudges = grudges if grudges is not None else self.grudges
        if all_grudges:
            latest = all_grudges[-1]
            bits.append(
                f"Unresolved: {latest.get('note', 'something')} — you have not let this go and "
                "it colors how you take their words."
            )
        if self.open_fight:
            bits.append(
                f"MIDDLE OF A FIGHT right now about: {self.open_fight.get('reason', 'something')}. "
                "You're not done. You are allowed to be hurt, to be brief, to be cold — "
                "but you're not allowed to be cruel for fun."
            )
        return " ".join(bits)
