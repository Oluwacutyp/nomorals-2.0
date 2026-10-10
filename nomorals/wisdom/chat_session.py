"""Chat delivery for WisdomKeeper practice sessions.

Terminal pacing (:meth:`PracticeGuide.run`) prints to a console the owner
may not be watching. This module delivers a session as timed chat
messages instead: SAFETY_TEXT first, then one message per phase, each
held for the phase's duration by a background pacer thread. Plain chat
words steer the run (pause/resume/stop) and the reply after a session
closes is journaled through :meth:`PracticeGuide.journal`.

Kernel invariant (same as practice.py): session pacing never calls
``time.sleep`` directly — it uses an injectable clock object with
``.sleep(seconds)`` and ``.now()`` (``RealClock`` in production, a fake
clock in tests). Pause *control* waits (the thread parking while the
user holds the session) use short real waits so a paused thread never
consumes fake-clock time.

Messaging reuses the existing stack: every send goes through
``Notifier.publish(..., critical=True, force=True)`` — critical because
a session the user explicitly started must complete even inside quiet
hours (like an alarm), force because phase messages sharing one run
title must not collapse into the dedupe window. No parallel messaging
stack is built here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

from ..core.ids import new_id
from .practice import RealClock

_log = logging.getLogger(__name__)

__all__ = [
    "ChatPracticeSession",
    "WisdomChatManager",
    "JOURNAL_PROMPT_TEMPLATE",
    "STATE_FILE_NAME",
]

#: on-disk handoff state, next to the practice log in the wisdom data dir.
STATE_FILE_NAME = "chat_state.json"

#: a phase sleep is chunked so stop/pause take effect within a few
#: seconds even on long phases (void-state has multi-minute phases).
SLEEP_CHUNK_SECONDS = 5.0

JOURNAL_PROMPT_TEMPLATE = (
    "\U0001f9d8 {name} {done}. How was that? What did you experience?\n"
    "(Reply here \u2014 I'll save it to your practice log.\n"
    "Add a 1\u20135 rating too if you like, e.g. \"4 calm and steady\".)"
)


class ChatPracticeSession:
    """One guided practice session delivered to chat as timed messages.

    Lifecycle: ``idle -> running -> (paused) -> completed | stopped``.
    ``start()`` validates the session id fail-fast (unknown id raises
    before any thread or message), then spawns a daemon pacer thread.
    ``join()`` waits for it.
    """

    #: plain chat words that steer a live run, matched on the whole
    #: lowercased message with edge punctuation stripped.
    PAUSE_WORDS = frozenset({"pause", "hold", "hold on", "wait"})
    RESUME_WORDS = frozenset({"resume", "continue", "go on", "unpause"})
    STOP_WORDS = frozenset({"stop", "cancel", "end", "quit"})

    ACTIVE_STATES = frozenset({"running", "paused", "stopping"})

    def __init__(self, guide: Any, session_id: str, *,
                 notifier: Any,
                 platform: str = "",
                 clock: Any = None,
                 midpoint_chime: bool = False) -> None:
        self.guide = guide
        self.session_id = session_id
        self._notifier = notifier
        self._platform = (platform or "").strip().lower()
        self._clock = clock if clock is not None else RealClock()
        # Midpoint chime ("halfway - settle deeper") at 50% of the plan.
        # Opt-in: the default phase stream is exactly the session script,
        # one message per phase, which existing consumers assert on.
        self._midpoint_chime = bool(midpoint_chime)
        self.run_id = new_id()
        self.state = "idle"
        self.on_finish: Callable[["ChatPracticeSession"], None] | None = None
        self._name = ""
        self._plan: list[tuple[str, float]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._hold = threading.Event()  # set while the thread parks on pause
        self._lock = threading.Lock()
        self._phases_done = 0
        self._midpoint_sent = False

    # ── introspection ─────────────────────────────────────────────
    @property
    def display_name(self) -> str:
        return self._name or self.session_id

    @property
    def is_active(self) -> bool:
        return self.state in self.ACTIVE_STATES

    @property
    def is_paused(self) -> bool:
        return self.state == "paused"

    def estimated_seconds(self) -> float:
        """Total paced seconds for this session (0 when not started)."""
        return sum(secs for _, secs in self._plan)

    def progress(self) -> dict[str, int]:
        """Live progress: phases done / total."""
        total = len(self._plan)
        with self._lock:
            done = self._phases_done
        return {"phases_done": done, "total": total}

    # ── control ───────────────────────────────────────────────────
    def start(self) -> None:
        """Validate + spawn the pacer thread. Raises PracticeError on an
        unknown session id — before any thread or message (fail fast)."""
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("chat practice session already started")
            # fail fast: raises PracticeError listing available sessions
            self._plan = self.guide.timed_phases_for_chat(self.session_id)
            names = {s["id"]: s["name"] for s in self.guide.list_sessions()}
            self._name = names.get(self.session_id, self.session_id)
            self._thread = threading.Thread(
                target=self._pump, name=f"wisdom-chat-{self.session_id}",
                daemon=True,
            )
            self._thread.start()

    def pause(self) -> bool:
        """Hold the run at the next phase boundary. False when not running."""
        with self._lock:
            if self.state != "running":
                return False
            self._paused.set()
            self._set_state_locked("paused")
            return True

    def resume(self) -> bool:
        """Continue a paused run. False when not paused."""
        with self._lock:
            if self.state != "paused":
                return False
            self._paused.clear()
            self._set_state_locked("running")
            return True

    def stop(self) -> bool:
        """End the run gracefully (journal prompt still goes out)."""
        with self._lock:
            if self.state not in self.ACTIVE_STATES:
                return False
            self._stop.set()
            self._paused.clear()  # unpark _wait_resumed so it can exit
            self._set_state_locked("stopping")
            return True

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the pacer thread. True when it finished."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _set_state_locked(self, state: str) -> None:
        self.state = state

    # ── the pacer ─────────────────────────────────────────────────
    def _pump(self) -> None:
        self._set_state_locked("running")
        phases_done = 0
        completed = False
        try:
            # SAFETY_TEXT first, verbatim, critical: an explicitly-started
            # session must complete even inside quiet hours (like an alarm).
            self._send(self.guide.safety_text(), critical=True)
            total = len(self._plan)
            midpoint = total // 2
            for msg, secs in self._plan:
                if self._stop.is_set():
                    break
                self._wait_resumed()
                if self._stop.is_set():
                    break
                # midpoint chime (the guided-session "settle deeper"
                # chapter mark): once, at 50% of the plan.
                if (self._midpoint_chime and total >= 4
                        and not self._midpoint_sent
                        and phases_done >= midpoint):
                    self._midpoint_sent = True
                    self._send("\u23f3 halfway \u2014 settle deeper.",
                               critical=True)
                self._send(msg, critical=True)
                phases_done += 1
                with self._lock:
                    self._phases_done = phases_done
                self._paced_sleep(secs)
            completed = not self._stop.is_set() and phases_done == total
        except Exception:  # noqa: BLE001 - a dead pacer must still journal-prompt
            _log.exception("chat practice session %r crashed", self.session_id)
        finally:
            try:
                self.guide.log_session(
                    self.session_id, completed=completed,
                    phases_done=phases_done)
            except Exception:  # noqa: BLE001 - logging must not kill delivery
                _log.warning("chat practice log_session failed", exc_info=True)
            self._set_state_locked("completed" if completed else "stopped")
            done_word = "complete" if completed else "ended early"
            self._send(
                JOURNAL_PROMPT_TEMPLATE.format(
                    name=self.display_name, done=done_word),
                critical=True,
            )
            callback = self.on_finish
        if callback is not None:
            try:
                callback(self)
            except Exception:  # noqa: BLE001
                _log.warning("chat practice on_finish callback failed",
                             exc_info=True)

    def _wait_resumed(self) -> None:
        """Park while paused. Real short waits — this is control plane,
        not session pacing, so the injectable clock stays untouched."""
        if not self._paused.is_set():
            return
        self._hold.set()
        try:
            while self._paused.is_set() and not self._stop.is_set():
                time.sleep(0.05)
        finally:
            self._hold.clear()

    def _paced_sleep(self, seconds: float) -> None:
        """Clock sleep in chunks so stop/pause land within seconds even
        on long phases. Pause is honored between chunks (a pause mid-phase
        holds the run instead of waiting out the whole phase)."""
        remaining = float(seconds)
        while remaining > 0:
            if self._stop.is_set():
                return
            self._wait_resumed()
            if self._stop.is_set():
                return
            chunk = min(remaining, SLEEP_CHUNK_SECONDS)
            self._clock.sleep(chunk)
            remaining -= chunk

    # ── delivery (existing stack only) ────────────────────────────
    def _send(self, text: str, *, critical: bool) -> bool:
        """One chat message via Notifier. Never raises — a failed send is
        logged and the pacer keeps the session's timing intact."""
        title = f"Wisdom practice \u00b7 {self.display_name}"
        try:
            result = self._notifier.publish(
                "wisdom-practice", title, text,
                critical=critical, force=True,
                channels=[self._platform] if self._platform else None,
            )
        except Exception:  # noqa: BLE001
            _log.warning("wisdom chat send failed", exc_info=True)
            return False
        return bool(result.get("delivered"))


class WisdomChatManager:
    """Owns chat practice sessions: one active run per chat key.

    Routes plain incoming words (pause/resume/stop) and the journal
    reply after a session closes. Journal-await state is persisted to
    ``chat_state.json`` so a CLI ``--chat`` handoff (or a restart) can
    leave the journal prompt behind; an *active* pacer thread cannot
    survive a process boundary and is dropped on load (documented,
    logged — never silently resumed).
    """

    def __init__(self, context: Any, *, clock: Any = None,
                 notifier: Any = None,
                 midpoint_chime: bool = False) -> None:
        self.context = context
        self._clock = clock
        self._notifier = notifier
        self._midpoint_chime = bool(midpoint_chime)
        self._guide: Any = None
        self._sessions: dict[str, ChatPracticeSession] = {}
        self._journal_await: dict[str, str] = {}  # chat_key -> session_id
        self._lock = threading.Lock()
        self._load_state()

    # ── lazy deps ─────────────────────────────────────────────────
    @property
    def _practice_guide(self) -> Any:
        if self._guide is None:
            from .practice import PracticeGuide
            self._guide = PracticeGuide(self.context)
        return self._guide

    @property
    def _live_notifier(self) -> Any:
        if self._notifier is None:
            from ..agents.notifier import Notifier
            self._notifier = Notifier(self.context)
        return self._notifier

    # ── sessions ──────────────────────────────────────────────────
    def start_session(self, chat_key: str, session_id: str,
                      *, platform: str = "",
                      midpoint_chime: bool | None = None) -> str:
        """Start a guided session in this chat. Returns the ack text for
        the chat reply. Raises PracticeError on an unknown session id
        (message lists available sessions); returns an error string when
        a session is already running here."""
        with self._lock:
            existing = self._sessions.get(chat_key)
            if existing is not None and existing.is_active:
                return (f"\u26a0\ufe0f a practice session is already running here "
                        f"({existing.display_name}) \u2014 reply stop to end it first.")
            session = ChatPracticeSession(
                self._practice_guide, session_id,
                notifier=self._live_notifier,
                platform=platform, clock=self._clock,
                midpoint_chime=(self._midpoint_chime
                                if midpoint_chime is None
                                else midpoint_chime),
            )
            session.on_finish = (
                lambda s, ck=chat_key: self._on_session_finish(ck, s))
            self._sessions[chat_key] = session
            try:
                session.start()  # fail fast on unknown id, before persist
            except Exception:
                self._sessions.pop(chat_key, None)
                raise
            self._persist()
            return (f"\U0001f9d8 starting {session.display_name} "
                    f"(\u2248{int(session.estimated_seconds())}s) \u2014 "
                    f"reply pause / resume / stop anytime.")

    def stop_session(self, chat_key: str) -> str:
        """End the active run in this chat, gracefully."""
        with self._lock:
            session = self._sessions.get(chat_key)
        if session is None or not session.is_active:
            return "no practice session is running here."
        session.stop()
        session.join(timeout=30)
        return f"\U0001f6d1 {session.display_name} stopped."

    def active_session(self, chat_key: str) -> ChatPracticeSession | None:
        return self._sessions.get(chat_key)

    def status(self, chat_key: str) -> dict[str, Any]:
        """JSON-able status for a chat: active run or journal-await."""
        session = self._sessions.get(chat_key)
        active = None
        if session is not None and session.is_active:
            active = {"session_id": session.session_id,
                      "name": session.display_name,
                      "state": session.state,
                      "progress": session.progress()}
        return {
            "chat_key": chat_key,
            "active": active,
            "journal_await": self._journal_await.get(chat_key),
        }

    def _on_session_finish(self, chat_key: str,
                           session: ChatPracticeSession) -> None:
        with self._lock:
            if self._sessions.get(chat_key) is session:
                del self._sessions[chat_key]
            # the journal prompt already went out; the next reply in this
            # chat is the journal entry.
            self._journal_await[chat_key] = session.session_id
            self._persist()

    # ── incoming words ────────────────────────────────────────────
    def handle_incoming(self, chat_key: str, text: str) -> str | None:
        """Route a plain incoming message. Returns the reply to send, or
        None when no session owns this message (normal flow continues).

        Only intercepts when a session is active in this chat or a
        journal reply is awaited; slash commands are never touched.
        """
        stripped = (text or "").strip()
        if not stripped or stripped.startswith("/"):
            return None
        word = " ".join(stripped.lower().split()).strip(".,!?;\u2026")
        with self._lock:
            session = self._sessions.get(chat_key)
            active = session is not None and session.is_active
            awaiting = self._journal_await.get(chat_key)
        if active:
            assert session is not None
            if word in ChatPracticeSession.STOP_WORDS:
                self.stop_session(chat_key)
                return "\U0001f6d1 session stopped."
            if word in ChatPracticeSession.PAUSE_WORDS:
                if session.pause():
                    return "\u23f8 paused \u2014 reply resume to continue."
                return "already paused."
            if word in ChatPracticeSession.RESUME_WORDS:
                if session.resume():
                    return "\u25b6 resuming."
                return "already running \u2014 reply pause to hold it."
            return None
        if awaiting:
            return self._take_journal(chat_key, awaiting, stripped)
        return None

    def _take_journal(self, chat_key: str, session_id: str,
                      notes: str) -> str:
        # A leading 1-5 ("4 calm and steady") is a post-session rating.
        rating: int | None = None
        rest = notes.strip()
        if rest[:1].isdigit():
            maybe = int(rest[:1])
            if 1 <= maybe <= 5 and (
                    len(rest) == 1 or not rest[1].isdigit()):
                rating = maybe
                rest = rest[1:].strip(" ,.:;\u2014-")
        journal_text = rest or notes.strip()
        try:
            self._practice_guide.journal(session_id, journal_text)
            if rating is not None:
                try:
                    self._practice_guide.rate(session_id, rating)
                except Exception:  # noqa: BLE001 - rating is bonus
                    _log.debug("wisdom rating store failed", exc_info=True)
                    rating = None
        except Exception as exc:  # noqa: BLE001 - fail fast, stay awaited
            _log.warning("wisdom journal store failed: %s", exc)
            return f"\u26a0\ufe0f couldn't save that ({exc}) \u2014 try again?"
        with self._lock:
            self._journal_await.pop(chat_key, None)
            self._persist()
        suffix = f" (rated {rating}\u2b50)" if rating else ""
        return f"\U0001f4dd saved to your practice log{suffix}. Nice work."

    # ── persistence ───────────────────────────────────────────────
    def _state_path(self):
        return self._practice_guide.data_dir() / STATE_FILE_NAME

    def _persist(self) -> None:
        try:
            state = {
                "journal_await": dict(self._journal_await),
                "active": {
                    ck: {"session_id": s.session_id, "state": s.state}
                    for ck, s in self._sessions.items() if s.is_active
                },
            }
            path = self._state_path()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(path)
        except Exception:  # noqa: BLE001 - state is best-effort
            _log.debug("wisdom chat state persist failed", exc_info=True)

    def _load_state(self) -> None:
        try:
            path = self._state_path()
        except Exception:  # noqa: BLE001
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:  # noqa: BLE001 - a torn state file must not break boot
            _log.warning("wisdom chat state unreadable, starting fresh",
                         exc_info=True)
            return
        for ck, sid in (raw.get("journal_await") or {}).items():
            self._journal_await[str(ck)] = str(sid)
        # active pacer threads cannot cross a process boundary: an entry
        # marked active here belongs to a dead process. Drop it loudly —
        # never silently resume a timed session nobody is pacing.
        for ck, info in (raw.get("active") or {}).items():
            _log.info("wisdom chat: dropping interrupted session %s (%s) "
                      "from a previous process", ck,
                      (info or {}).get("session_id", "?"))

    # ── shutdown ──────────────────────────────────────────────────
    def shutdown(self) -> None:
        """Stop every live pacer (clean shutdown). In-memory only — the
        journal-await file stays so replies can still be journaled."""
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            try:
                session.stop()
            except Exception:  # noqa: BLE001
                pass
        for session in sessions:
            try:
                session.join(timeout=10)
            except Exception:  # noqa: BLE001
                pass
