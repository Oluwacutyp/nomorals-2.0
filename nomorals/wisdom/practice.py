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

# ── safety framing ────────────────────────────────────────────────────
# The five required qualifications. Printed at the start of every run
# and returned by PracticeGuide.safety_text(). Plain statements, no
# refusal machinery.
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

    def _log_path(self) -> Path:
        return self._wisdom_root() / "practice_log.jsonl"

    # ── loading ───────────────────────────────────────────────────────
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
        if not sessions:
            raise PracticeError(
                f"no session scripts found in {self._sessions_dir}")
        return sessions

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
        session = self._get(session_id)
        messages: list[str] = []
        for phase in session.phases:
            secs = phase.seconds
            secs_s = str(int(secs)) if float(secs).is_integer() else str(secs)
            for _ in range(phase.repeat):
                messages.append(f"{phase.label} ({secs_s}s): {phase.instruction}")
        return messages

    # ── run ───────────────────────────────────────────────────────────
    def run(self, session_id: str, *, clock: Any = None,
            out: Callable[[str], None] | None = None,
            rounds: int | None = None) -> dict[str, Any]:
        """Pace a session. Prints each phase label + instruction, counts
        down, then moves on. ``clock`` must provide ``.sleep(seconds)``
        and ``.now() -> float``. ``rounds`` repeats the whole phase
        sequence that many times."""
        session = self._get(session_id)
        if rounds is None:
            rounds = 1
        if isinstance(rounds, bool) or not isinstance(rounds, int) \
                or rounds < 1:
            raise PracticeError(
                f"rounds must be a positive int, got {rounds!r}")
        clock = clock if clock is not None else RealClock()
        out = out if out is not None else print

        started_at = datetime.fromtimestamp(
            clock.now(), tz=timezone.utc).isoformat()

        out(f"=== {session.name} ===")
        out(self.safety_text())
        out("")

        plan: list[Phase] = []
        for _ in range(rounds):
            for phase in session.phases:
                plan.extend([phase] * phase.repeat)
        total = len(plan)

        phases_done = 0
        for i, phase in enumerate(plan, 1):
            out(f"[{i}/{total}] {phase.label}: {phase.instruction}")
            self._countdown(phase.seconds, out, clock)
            phases_done += 1

        out(f"Done — {session.name} complete.")

        entry = {
            "type": "session",
            "session_id": session.id,
            "started_at": started_at,
            "completed": True,
            "phases_done": phases_done,
            "notes": "",
        }
        self._append_log(entry)
        return {
            "session_id": session.id,
            "started_at": started_at,
            "completed": True,
            "phases_done": phases_done,
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
    def _append_log(self, entry: dict[str, Any]) -> None:
        path = self._log_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
