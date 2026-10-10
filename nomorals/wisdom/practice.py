"""PracticeGuide — data-driven guided breathing & sitting sessions.

Session scripts are JSON *data* in ``nomorals/wisdom/sessions/``; this
module is only the pacer that reads them. Every run prints the safety
qualifications first: slow breathing with a longer exhale relaxes via
the parasympathetic system, it guarantees no particular state, stop if
dizzy or uncomfortable, clinician check for lung/heart/pregnancy/BP
conditions, and this is not medical or mental-health care.

Kernel invariant: session pacing never calls ``time.sleep`` directly.
It uses an injectable clock object with ``.sleep(seconds)`` and
``.now()`` (defaults to :class:`RealClock`); tests pass a fake clock.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .errors import PracticeError
from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break a practice run (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

# ── safety framing ────────────────────────────────────────────────────
# Safety notes. Shown on request via PracticeGuide.safety_text(), or once
# per session if the owner enables it (show_safety=True). Not printed on
# every run by default.
SAFETY_TEXT = (
    "Safety notes for breathing practice:\n"
    "(1) Slow breathing with a longer exhale promotes relaxation via the "
    "parasympathetic nervous system — this is well supported.\n"
    "(2) It does NOT guarantee any particular state — there is no verified "
    "link between breathing techniques and astral projection or kundalini "
    "awakening.\n"
    "(3) If you feel lightheaded, dizzy, or uncomfortable, stop and breathe "
    "normally.\n"
    "(4) If you have a lung or heart condition, are pregnant, or have "
    "blood-pressure conditions, check with a clinician first.\n"
    "(5) This is not medical or mental-health care."
)

# Phase instructions must stay short and self-contained so a chat bot
# can send them one by one as timed messages.
MAX_INSTRUCTION_CHARS = 200

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class RealClock:
    """Default clock: the real one. This is the ONLY ``time.sleep`` in
    this module — session logic below never sleeps directly."""

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def now(self) -> float:
        return time.time()


# ── session data model ────────────────────────────────────────────────

@dataclass
class Phase:
    """One timed step of a session script."""
    label: str
    seconds: float
    instruction: str
    repeat: int = 1

    @classmethod
    def from_dict(cls, d: Any, source: str) -> "Phase":
        if not isinstance(d, dict):
            raise PracticeError(f"malformed phase in {source}: not an object")
        try:
            phase = cls(
                label=str(d["label"]),
                seconds=d["seconds"],
                instruction=str(d["instruction"]),
                repeat=int(d.get("repeat", 1)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PracticeError(
                f"malformed phase in {source}: {exc}") from exc
        phase.validate(source)
        return phase

    def validate(self, source: str) -> None:
        if not self.label or not isinstance(self.label, str):
            raise PracticeError(f"phase in {source} has no label")
        if (isinstance(self.seconds, bool)
                or not isinstance(self.seconds, (int, float))
                or self.seconds <= 0):
            raise PracticeError(
                f"phase {self.label!r} in {source} has bad seconds: "
                f"{self.seconds!r}")
        if not self.instruction or not isinstance(self.instruction, str):
            raise PracticeError(
                f"phase {self.label!r} in {source} has no instruction")
        if len(self.instruction) > MAX_INSTRUCTION_CHARS:
            raise PracticeError(
                f"phase {self.label!r} in {source} instruction exceeds "
                f"{MAX_INSTRUCTION_CHARS} chars "
                f"({len(self.instruction)} given)")
        if (isinstance(self.repeat, bool)
                or not isinstance(self.repeat, int) or self.repeat < 1):
            raise PracticeError(
                f"phase {self.label!r} in {source} has bad repeat: "
                f"{self.repeat!r}")

    def total_seconds(self) -> float:
        return self.seconds * self.repeat


@dataclass
class SessionScript:
    """One validated session script loaded from JSON data."""
    id: str
    name: str
    description: str
    safety_note_id: str
    beginner: bool
    phases: list[Phase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Any, source: str) -> "SessionScript":
        if not isinstance(d, dict):
            raise PracticeError(f"malformed session in {source}: not an object")
        try:
            script = cls(
                id=str(d["id"]),
                name=str(d["name"]),
                description=str(d["description"]),
                safety_note_id=str(d["safety_note_id"]),
                beginner=d["beginner"],
                phases=[Phase.from_dict(p, source) for p in d["phases"]],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PracticeError(
                f"malformed session in {source}: {exc}") from exc
        script.validate(source)
        return script

    def validate(self, source: str) -> None:
        if not self.id or not _ID_RE.match(self.id):
            raise PracticeError(f"session in {source} has bad id: {self.id!r}")
        if not self.name:
            raise PracticeError(f"session {self.id!r} has no name")
        if not self.description:
            raise PracticeError(f"session {self.id!r} has no description")
        if not self.safety_note_id:
            raise PracticeError(f"session {self.id!r} has no safety_note_id")
        if not isinstance(self.beginner, bool):
            raise PracticeError(
                f"session {self.id!r} has non-bool beginner: "
                f"{self.beginner!r}")
        if not self.phases:
            raise PracticeError(f"session {self.id!r} has no phases")

    def total_seconds(self) -> float:
        return sum(p.total_seconds() for p in self.phases)


# ── PracticeGuide ─────────────────────────────────────────────────────

class PracticeGuide:
    """Paces data-driven practice sessions.

    ``PracticeGuide(context)`` loads the JSON session scripts shipped in
    the package ``sessions/`` dir. Pass ``clock`` (with ``.sleep`` and
    ``.now``) to :meth:`run` to control time in tests.
    """

    def __init__(self, context: Any, sessions_dir: Path | str | None = None
                 ) -> None:
        self.context = context
        if sessions_dir is None:
            sessions_dir = Path(__file__).resolve().parent / "sessions"
        self._sessions_dir = Path(sessions_dir)
        self._sessions = self._load_sessions()

    # ── paths ─────────────────────────────────────────────────────────
    def _wisdom_root(self) -> Path:
        settings = getattr(self.context, "settings", None)
        root = getattr(settings, "workspace_dir", None) \
            if settings is not None else None
        d = Path(root) / "wisdom" if root else Path.cwd() / "workspace" / "wisdom"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def data_dir(self) -> Path:
        """Workspace dir for wisdom state (practice log, chat state)."""
        return self._wisdom_root()

    def _log_path(self) -> Path:
        return self._wisdom_root() / "practice_log.jsonl"

    # ── loading ───────────────────────────────────────────────────────
    def _custom_dir(self) -> Path:
        """Workspace dir for user-created sessions. Package sessions are
        read-only data; custom programs live here so upgrades never
        clobber them."""
        d = self._wisdom_root() / "custom_sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_sessions(self) -> dict[str, SessionScript]:
        if not self._sessions_dir.is_dir():
            raise PracticeError(
                f"sessions dir missing: {self._sessions_dir}")
        sessions: dict[str, SessionScript] = {}
        for path in sorted(self._sessions_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PracticeError(
                    f"could not parse session file {path.name}: {exc}"
                ) from exc
            script = SessionScript.from_dict(raw, source=path.name)
            if script.id in sessions:
                raise PracticeError(
                    f"duplicate session id {script.id!r} in {path.name}")
            sessions[script.id] = script
        # User-created programs overlay the packaged set. A custom id
        # that collides with a packaged session is rejected at creation
        # time (create_custom), so this merge is collision-free.
        custom_dir = self._wisdom_root() / "custom_sessions"
        if custom_dir.is_dir():
            for path in sorted(custom_dir.glob("*.json")):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue  # a torn custom file must not break boot
                try:
                    script = SessionScript.from_dict(
                        raw, source=f"custom/{path.name}")
                except PracticeError:
                    continue
                sessions[script.id] = script
        if not sessions:
            raise PracticeError(
                f"no session scripts found in {self._sessions_dir}")
        return sessions

    def create_custom(self, session_id: str, name: str, description: str,
                      phases: list[dict[str, Any]], *,
                      beginner: bool = True,
                      safety_note_id: str = "breathwork-general"
                      ) -> SessionScript:
        """Create and save a custom breathing program (the Awesome
        Breathing / Paced Breathing "custom program" feature).

        ``phases`` is a list of {"label", "seconds", "instruction",
        "repeat"?} dicts — the same schema as the packaged JSON.
        Validated exactly like packaged sessions; the id must be new
        (never shadows a packaged session). Persisted to the workspace
        custom_sessions dir and live immediately.
        """
        payload = {
            "id": session_id,
            "name": name,
            "description": description,
            "safety_note_id": safety_note_id,
            "beginner": beginner,
            "phases": phases,
        }
        script = SessionScript.from_dict(payload, source="custom-request")
        if script.id in self._sessions:
            raise PracticeError(
                f"session id {script.id!r} already exists; pick another")
        path = self._custom_dir() / f"{script.id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        self._sessions[script.id] = script
        _emit("wisdom.practice.custom_created", {
            "session_id": script.id, "name": script.name,
            "total_seconds": script.total_seconds(),
        })
        return script

    def delete_custom(self, session_id: str) -> bool:
        """Delete a user-created program. Packaged sessions are
        read-only and cannot be deleted. True when one was removed."""
        session = self._get(session_id)
        path = self._custom_dir() / f"{session.id}.json"
        if not path.is_file():
            raise PracticeError(
                f"session {session_id!r} is a packaged session and "
                f"cannot be deleted")
        path.unlink()
        del self._sessions[session.id]
        return True

    def _get(self, session_id: str) -> SessionScript:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise PracticeError(
                f"unknown session {session_id!r}; available: "
                f"{', '.join(sorted(self._sessions))}") from None

    # ── safety ────────────────────────────────────────────────────────
    def safety_text(self) -> str:
        """The five safety qualifications, returned verbatim."""
        return SAFETY_TEXT

    # ── catalog ───────────────────────────────────────────────────────
    def list_sessions(self) -> list[dict[str, Any]]:
        """id, name, description, beginner flag, total seconds."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "beginner": s.beginner,
                "total_seconds": s.total_seconds(),
            }
            for s in sorted(self._sessions.values(), key=lambda s: s.id)
        ]

    def phases_for_chat(self, session_id: str) -> list[str]:
        """Flat per-phase message strings for chat delivery: short,
        self-contained, one per timed message."""
        return [msg for msg, _ in self.timed_phases_for_chat(session_id)]

    def timed_phases_for_chat(self, session_id: str) -> list[tuple[str, float]]:
        """Like :meth:`phases_for_chat` but each message is paired with
        its phase's seconds, for timed chat delivery. Raises
        PracticeError on an unknown session id (fail fast)."""
        session = self._get(session_id)
        plan: list[tuple[str, float]] = []
        for phase in session.phases:
            secs = phase.seconds
            secs_s = str(int(secs)) if float(secs).is_integer() else str(secs)
            for _ in range(phase.repeat):
                plan.append(
                    (f"{phase.label} ({secs_s}s): {phase.instruction}",
                     float(secs)))
        return plan

    # ── run ───────────────────────────────────────────────────────────
    def run(self, session_id: str, *, clock: Any = None,
            out: Callable[[str], None] | None = None,
            rounds: int | None = None,
            show_safety: bool = False,
            settle_seconds: float = 0.0,
            ramp: float = 1.0) -> dict[str, Any]:
        """Pace a session. Prints each phase label + instruction, counts
        down, then moves on. ``clock`` must provide ``.sleep(seconds)``
        and ``.now() -> float``. ``rounds`` repeats the whole phase
        sequence that many times. ``show_safety`` prints safety notes first.

        ``settle_seconds`` is the pre-session settle-in countdown (the
        "few moments to settle in" the best breathing apps offer).
        ``ramp`` gradually scales breath times across rounds: 1.0 is
        flat, 1.5 ends rounds 50% longer than they start (Paced
        Breathing's ramp mode). A midpoint chime goes out at 50%.
        """
        session = self._get(session_id)
        if rounds is None:
            rounds = 1
        if isinstance(rounds, bool) or not isinstance(rounds, int) \
                or rounds < 1:
            raise PracticeError(
                f"rounds must be a positive int, got {rounds!r}")
        if isinstance(settle_seconds, bool) or not isinstance(
                settle_seconds, (int, float)) or settle_seconds < 0:
            raise PracticeError(
                f"settle_seconds must be >= 0, got {settle_seconds!r}")
        if isinstance(ramp, bool) or not isinstance(
                ramp, (int, float)) or not 0.25 <= ramp <= 3.0:
            raise PracticeError(
                f"ramp must be a number in [0.25, 3.0], got {ramp!r}")
        clock = clock if clock is not None else RealClock()
        out = out if out is not None else print

        started_at = datetime.fromtimestamp(
            clock.now(), tz=timezone.utc).isoformat()

        # The paced plan: (phase, effective_seconds) per execution, with
        # the ramp factor interpolated linearly across rounds.
        plan: list[tuple[Phase, float]] = []
        for r in range(rounds):
            factor = 1.0 + (ramp - 1.0) * (r / max(1, rounds - 1))
            for phase in session.phases:
                for _ in range(phase.repeat):
                    plan.append((phase, phase.seconds * factor))
        total = len(plan)
        paced_seconds = sum(secs for _, secs in plan)

        _emit("wisdom.practice.started", {
            "session_id": session.id,
            "name": session.name,
            "rounds": rounds,
            "ramp": ramp,
            "total_seconds": paced_seconds,
        })

        out(f"=== {session.name} ===")
        if show_safety:
            out(self.safety_text())
            out("")
        if settle_seconds > 0:
            out(f"Settle in… ({settle_seconds:g}s)")
            self._countdown(float(settle_seconds), out, clock)

        midpoint = total // 2 + 1
        phases_done = 0
        for i, (phase, secs) in enumerate(plan, 1):
            if total >= 4 and i == midpoint:
                out("— halfway — settle deeper —")
            out(f"[{i}/{total}] {phase.label}: {phase.instruction}")
            self._countdown(secs, out, clock)
            phases_done += 1

        out(f"Done — {session.name} complete.")

        entry = {
            "type": "session",
            "session_id": session.id,
            "started_at": started_at,
            "completed": True,
            "phases_done": phases_done,
            "notes": "",
            "rounds": rounds,
            "ramp": ramp,
            "total_seconds": paced_seconds,
        }
        self._append_log(entry)
        _emit("wisdom.practice.completed", {
            "session_id": session.id,
            "name": session.name,
            "started_at": started_at,
            "completed": True,
            "phases_done": phases_done,
            "total_seconds": paced_seconds,
        })
        return {
            "session_id": session.id,
            "started_at": started_at,
            "completed": True,
            "phases_done": phases_done,
            "total_seconds": paced_seconds,
        }

    def _countdown(self, seconds: float, out: Callable[[str], None],
                   clock: Any) -> None:
        whole = int(seconds)
        for remaining in range(whole, 0, -1):
            out(f"  {remaining}…")
            clock.sleep(1)
        frac = seconds - whole
        if frac > 0:
            clock.sleep(frac)

    # ── journal ───────────────────────────────────────────────────────
    def journal(self, session_id: str, notes: str) -> dict[str, Any]:
        """Append post-session notes to the practice log. Empty notes are
        rejected fail-fast — nothing to store."""
        session = self._get(session_id)
        text = (notes or "").strip()
        if not text:
            raise PracticeError("journal notes are empty — nothing to store")
        entry = {
            "type": "journal",
            "session_id": session.id,
            "journaled_at": datetime.now(timezone.utc).isoformat(),
            "notes": text,
        }
        self._append_log(entry)
        return entry

    def log_session(self, session_id: str, *, completed: bool,
                    phases_done: int, notes: str = "") -> dict[str, Any]:
        """Record a chat-driven session run in the practice log, using the
        same ``session`` entry schema as :meth:`run`. Raises PracticeError
        on an unknown session id (fail fast)."""
        session = self._get(session_id)
        entry = {
            "type": "session",
            "session_id": session.id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed": bool(completed),
            "phases_done": int(phases_done),
            "notes": notes,
            "via": "chat",
        }
        self._append_log(entry)
        return entry

    # ── history ───────────────────────────────────────────────────────
    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent practice log entries, newest first."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise PracticeError(
                f"history limit must be a positive int, got {limit!r}")
        path = self._log_path()
        if not path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue  # a log file must survive a torn write
        return entries[-limit:][::-1]

    # ── log plumbing ──────────────────────────────────────────────────
    # -- ratings --------------------------------------------------
    def rate(self, session_id: str, stars: int) -> dict[str, Any]:
        """Record a 1-5 post-session rating. Fail fast on unknown
        sessions and out-of-range stars."""
        session = self._get(session_id)
        if isinstance(stars, bool) or not isinstance(stars, int) \
                or not 1 <= stars <= 5:
            raise PracticeError(
                f"rating must be an int 1-5, got {stars!r}")
        entry = {
            "type": "rating",
            "session_id": session.id,
            "rated_at": datetime.now(timezone.utc).isoformat(),
            "stars": stars,
        }
        self._append_log(entry)
        _emit("wisdom.practice.rated", {
            "session_id": session.id, "stars": stars})
        return entry

    # -- stats ------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Practice stats: sessions, minutes, streaks, ratings.

        The streak feature the best breathing apps ship (Paced
        Breathing PB+): consecutive UTC days with at least one
        completed session, counting back from today.
        """
        entries = self.history(limit=100000)
        sessions = [e for e in entries
                    if e.get("type") == "session"]
        completed = [e for e in sessions if e.get("completed")]
        total_seconds = sum(float(e.get("total_seconds") or 0)
                            for e in completed)
        by_session: dict[str, int] = {}
        for e in completed:
            sid = str(e.get("session_id", "?"))
            by_session[sid] = by_session.get(sid, 0) + 1
        # streak: consecutive UTC days with >= 1 completed session
        days: set[str] = set()
        for e in completed:
            try:
                day = datetime.fromisoformat(
                    str(e["started_at"])).date().isoformat()
            except (KeyError, ValueError):
                continue
            days.add(day)
        streak = 0
        cursor = datetime.now(timezone.utc).date()
        if cursor.isoformat() not in days:
            # allow the streak to survive "today not yet practiced"
            cursor = cursor.fromordinal(cursor.toordinal() - 1)
        while cursor.isoformat() in days:
            streak += 1
            cursor = cursor.fromordinal(cursor.toordinal() - 1)
        ratings: dict[str, list[int]] = {}
        for e in entries:
            if e.get("type") == "rating":
                try:
                    stars = int(e["stars"])
                except (KeyError, TypeError, ValueError):
                    continue
                ratings.setdefault(
                    str(e.get("session_id", "?")), []).append(stars)
        avg_rating = {sid: round(sum(v) / len(v), 2)
                      for sid, v in ratings.items()}
        last = completed[0] if completed else None
        return {
            "sessions_total": len(sessions),
            "sessions_completed": len(completed),
            "minutes_practiced": round(total_seconds / 60.0, 1),
            "streak_days": streak,
            "by_session": dict(sorted(by_session.items())),
            "avg_rating": avg_rating,
            "last_session": ({
                "session_id": last.get("session_id"),
                "started_at": last.get("started_at"),
            } if last else None),
        }

    # -- catalog presentation ---------------------------------------
    @staticmethod
    def _rhythm_glyph(label: str) -> str:
        low = (label or "").lower()
        if low.startswith("inhale"):
            return "\u25b2"  # ▲
        if low.startswith("exhale"):
            return "\u25bc"  # ▼
        if low.startswith("hold"):
            return "\u25a0"  # ■
        return "\u25cf"  # ●

    def rhythm_signature(self, session_id: str) -> str:
        """Compact pattern signature, e.g. "4-4-4-4" for box breathing:
        the seconds of one phase cycle, joined."""
        session = self._get(session_id)
        parts = []
        for phase in session.phases:
            secs = phase.seconds
            parts.append(str(int(secs)) if float(secs).is_integer()
                         else str(secs))
        return "-".join(parts)

    def catalog_text(self) -> str:
        """God-tier session catalog: rhythm glyphs + pattern signature
        + duration, not a bare id list."""
        lines = ["\U0001f9d8 Practice sessions", ""]
        for info in self.list_sessions():
            session = self._get(info["id"])
            glyphs = " ".join(
                self._rhythm_glyph(ph.label) for ph in session.phases)
            sig = self.rhythm_signature(info["id"])
            mins = info["total_seconds"] / 60.0
            beginner = "beginner-friendly" if info["beginner"] else "advanced"
            lines.append(f"{glyphs}  {info['name']} [{beginner}]")
            lines.append(f"    {info['description']}")
            lines.append(
                f"    pattern {sig} \u00b7 \u2248{mins:.0f} min "
                f"\u00b7 id: {info['id']}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _append_log(self, entry: dict[str, Any]) -> None:
        path = self._log_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
