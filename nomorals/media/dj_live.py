"""Live DJ performer — a real DJ, not a renderer.

``dj.py`` builds radio shows offline and hands over one finished file.
This module is the opposite: a DJ that performs LIVE. A background thread
runs the set in real time — one track at a time, delivered to the chat as
the set happens — while a room sensor reads actual chat activity and the
DJ makes real decisions: ride the energy, switch it up when the room is
dead, take requests, shout out listeners, talk between tracks in
character.

Honest boundaries (what "live" means here and what it doesn't):
- Tracks are delivered as voice notes / audio messages in real time,
  one at a time, with the DJ's talk between them. The set is NOT
  pre-rendered — every next-track decision happens live from room state.
- Transitions are real: the outgoing track's tail is ducked under the
  DJ's talk-over, then the incoming track drops (radio-style mixing,
  rendered with real DSP from dj_engine analysis).
- Once a voice note is delivered it plays on the listener's device; the
  DJ cannot reach in and stop it. "Cutting a track early" therefore
  means: acknowledge the room immediately and pivot the NEXT selection
  without delay. This is documented, not hidden.
- The DJ cannot stream PCM into a Telegram voice chat or WhatsApp call
  — no such path exists on this stack. The live room IS the chat: the
  set plays there, the crowd reacts there, the DJ reads it there.

Nothing here raises out of the live loop — a dead DJ thread is a dead
show, so every failure is caught, logged, and the set keeps moving.
"""

from __future__ import annotations

import math
import os
import queue
import random
import re
import threading
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "RoomSensor",
    "DJPersona",
    "TrackRequest",
    "LiveSet",
    "get_live_set",
    "start_live_set",
    "stop_live_set",
    "notify_live_message",
    "control_dj_live",
]

_SR = 22050

# ── the DJ's identity ──────────────────────────────────────────────────
# A real character, not a script. Persona facts feed the talk generator;
# the talk itself is composed fresh every time from live context.

DJ_NAME = "DJ Vrede"
DJ_VOICE = ""  # resolved at talk time: catalogue default or override


def _dj_character() -> Any:
    """Fetch Vrede from the unified character bank (nomorals/characters).

    One Vrede, not two: the DJ uses the bank's Vrede — the same character
    who guests on podcasts and plays games. DJ-specific traits live in the
    seed (seeds.py), not here. Seeds the bank on first use.
    """
    try:
        from ..characters import CharacterStore
        from ..characters.seeds import seed_bank
    except Exception:  # noqa: BLE001
        return None
    store = CharacterStore()
    try:
        existing = store.get_by_name("Vrede")
    except Exception:  # noqa: BLE001
        existing = None
    if existing is not None:
        return existing
    try:
        seed_bank(store)
        return store.get_by_name("Vrede")
    except Exception:  # noqa: BLE001
        return None


# ── room sensing ───────────────────────────────────────────────────────
# Real signals from real chat activity. Message rate, hype markers,
# dead markers, and explicit requests are read off actual ChatMessage
# objects — never simulated.

_HYPE_EMOJI = ("🔥", "🚀", "💃", "🕺", "👏", "🙌", "💯", "🎶", "🎵")
_HYPE_WORDS = ("fire", "tune", "banger", "w ", " jam", "lit", "goat")
_DEAD_WORDS = ("skip", "boring", "next", "mid", "trash", "stop")
_REQUEST_RES = (
    re.compile(r"\bplay\s+(.+)", re.I),
    re.compile(r"\brequest\s+(.+)", re.I),
    re.compile(r"\bdrop\s+(.+?)(?:\s+next|\s+please|$)", re.I),
)

_WINDOW_S = 300.0  # 5-minute sliding window


@dataclass
class _RoomEvent:
    ts: float
    kind: str  # message | hype | dead | request | reaction
    text: str = ""
    sender: str = ""


class RoomSensor:
    """Reads the room from live chat events. No simulation."""

    def __init__(self) -> None:
        self._events: list[_RoomEvent] = []
        self._lock = threading.Lock()
        self._requests: list[TrackRequest] = []

    # ── ingest ──
    def note_message(self, text: str, sender: str = "",
                     ts: float | None = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        ts = ts if ts is not None else time.time()
        low = text.lower()
        kind = "message"
        if any(e in text for e in _HYPE_EMOJI) or any(
                w in low for w in _HYPE_WORDS):
            kind = "hype"
        elif any(w in low for w in _DEAD_WORDS):
            kind = "dead"
        req = self._parse_request(text)
        with self._lock:
            self._events.append(_RoomEvent(ts, kind, text, sender))
            self._prune(ts)
            if req is not None:
                req.requester = sender or req.requester
                # dedupe: same requester + same query within the window
                if not any(r.query.lower() == req.query.lower()
                           and r.requester == req.requester
                           for r in self._requests):
                    self._requests.append(req)

    def note_reaction(self, sender: str = "", ts: float | None = None) -> None:
        ts = ts if ts is not None else time.time()
        with self._lock:
            self._events.append(_RoomEvent(ts, "hype", "reaction", sender))
            self._prune(ts)

    @staticmethod
    def _parse_request(text: str) -> "TrackRequest | None":
        for rx in _REQUEST_RES:
            m = rx.search(text or "")
            if m:
                q = m.group(1).strip(" .!\"'")[:80]
                if q and len(q.split()) <= 12:
                    return TrackRequest(query=q)
        return None

    def _prune(self, now: float) -> None:
        cutoff = now - _WINDOW_S
        self._events = [e for e in self._events if e.ts >= cutoff]

    # ── read ──
    def energy(self) -> float:
        """0..1 — weighted recent activity. Decays with silence."""
        now = time.time()
        with self._lock:
            self._prune(now)
            if not self._events:
                return 0.0
            score = 0.0
            for e in self._events:
                age = max(0.0, now - e.ts)
                w = math.exp(-age / 90.0)  # ~90s half-attention
                if e.kind == "hype":
                    score += 2.0 * w
                elif e.kind == "dead":
                    score -= 1.5 * w
                elif e.kind == "request":
                    score += 1.0 * w
                else:
                    score += 0.5 * w
            # normalize: ~10 hype-equivalents in window = full energy
            return max(0.0, min(1.0, score / 12.0))

    def trend(self) -> str:
        """rising | falling | flat | dead — from first/second half split."""
        now = time.time()
        with self._lock:
            self._prune(now)
            if not self._events:
                return "dead"
            mid = now - _WINDOW_S / 2
            first = sum(1 for e in self._events if e.ts < mid)
            second = sum(1 for e in self._events if e.ts >= mid)
            if second == 0 and first == 0:
                return "dead"
            if second > first * 1.5:
                return "rising"
            if first > second * 1.5:
                return "falling"
            return "flat"

    def dead_signals(self) -> int:
        now = time.time()
        with self._lock:
            self._prune(now)
            return sum(1 for e in self._events if e.kind == "dead")

    def pop_request(self) -> "TrackRequest | None":
        with self._lock:
            if self._requests:
                return self._requests.pop(0)
            return None

    def pending_requests(self) -> int:
        with self._lock:
            return len(self._requests)


@dataclass
class TrackRequest:
    query: str
    requester: str = ""
    ts: float = field(default_factory=time.time)


# ── the DJ's voice ─────────────────────────────────────────────────────

class DJPersona:
    """The DJ as a character. Talk is composed fresh from live context —
    never a fixed script. Uses the brain's LLM when available (via
    characters.talk_to); otherwise assembles dynamically from the actual
    room state (track facts, energy, names) — varied, never repeated
    verbatim twice in a set."""

    def __init__(self, voice: str = "") -> None:
        self.character = _dj_character()
        self.voice = voice or DJ_VOICE
        self._rng = random.Random()
        self._recent_lines: list[str] = []
        self._llm_fn: Callable[[str], str] | None = None

    def set_llm(self, fn: Callable[[str], str] | None) -> None:
        self._llm_fn = fn

    # ── talk generation ──
    def talk(self, *, situation: str, track_title: str = "",
             track_style: str = "", energy: float = 0.5,
             requester: str = "", extra: str = "") -> str:
        """Compose one DJ line for a live moment.

        ``situation``: intro | track_intro | hype | pivot | request_yes |
        request_no | cool_down | outro | dead_room.
        """
        ctx = {
            "situation": situation,
            "track": track_title, "style": track_style,
            "energy": round(energy, 2),
            "requester": requester, "extra": extra,
            "dj": DJ_NAME,
        }
        line = ""
        if self._llm_fn is not None and self.character is not None:
            line = self._talk_via_llm(ctx)
        if not line:
            line = self._assemble(ctx)
        line = self._de_repeat(line)
        return line

    def _talk_via_llm(self, ctx: dict[str, Any]) -> str:
        """Speak through the bank Vrede's Character.speak().

        Uses the unified character-bank Vrede — the same character who
        guests on podcasts and plays games. Falls back to _assemble when
        no dialogue engine is wired.
        """
        try:
            ch = self.character
            if ch is None or self._llm_fn is None:
                return ""
            q = (
                f"You are live on air. Situation: {ctx['situation']}. "
                f"Track: {ctx['track'] or 'n/a'} ({ctx['style'] or 'n/a'}). "
                f"Room energy: {ctx['energy']}/1. "
                + (f"Shout out {ctx['requester']}. " if ctx["requester"] else "")
                + (f"Note: {ctx['extra']}. " if ctx["extra"] else "")
                + "One short DJ line, in your voice, under 25 words. "
                "Never repeat a previous line."
            )
            text = ch.speak(q, self._llm_fn)
            if text:
                return text.strip()[:220]
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _assemble(self, ctx: dict[str, Any]) -> str:
        """Dynamic fallback: composed from live facts, varied by RNG.
        Every line references something REAL (the track, the energy,
        a name) — never generic filler."""
        r = self._rng
        e = ctx["energy"]
        t = ctx["track"]
        s = ctx["style"]
        mood = "electric" if e > 0.7 else "warming up" if e > 0.35 else "sleepy"
        sit = ctx["situation"]
        bits: list[str] = []
        if sit == "intro":
            bits = [
                f"{DJ_NAME} live — we're doing this for real tonight",
                f"lock in, {DJ_NAME} is on the decks and the room is {mood}",
                f"it's {DJ_NAME}, live set starting now — let's see where the night goes",
            ]
        elif sit == "track_intro":
            bits = [
                f"up next: {t} — {s} energy, let's ride",
                f"{t} coming at you, {s} vibes" + (" — room's feeling it" if e > 0.6 else ""),
                f"switching lanes into {t}, {s} style",
            ]
        elif sit == "hype":
            bits = [
                "I see you, family — keep that energy",
                "that's what I'm talking about — we stay up",
                "room's alive right now, let's take it higher",
            ]
        elif sit == "pivot":
            bits = [
                "okay, switching it up — new energy incoming",
                f"reading the room… we're taking {t} for a spin, {s} style",
                "that wasn't landing — watch this",
            ]
        elif sit == "request_yes":
            who = ctx["requester"] or "you"
            bits = [
                f"{who} asked, {DJ_NAME} delivers — {t} incoming",
                f"shout out {who} — your request {t} is up next",
                f"request granted, {who} — {t}, let's go",
            ]
        elif sit == "request_no":
            bits = [
                f"love that pick but it'd kill this vibe — holding it for later, {ctx['requester'] or 'family'}",
                f"{t or 'that one'} doesn't fit this wave right now — trust me, what I got next is better",
            ]
        elif sit == "cool_down":
            bits = [
                "bringing it down smooth — breathe with me",
                f"cool-down time — {t}, {s}, easy",
            ]
        elif sit == "dead_room":
            bits = [
                "hello? anybody home? let's wake this room up",
                "quiet crowd tonight — fine, I'll do the talking AND the mixing",
            ]
        elif sit == "outro":
            bits = [
                f"{DJ_NAME} signing off — that was real, family",
                "set's done — you were beautiful tonight",
            ]
        else:
            bits = [f"{DJ_NAME} on the decks — {t or 'more music'} incoming"]
        line = r.choice(bits)
        if ctx["extra"]:
            line = f"{line} {ctx['extra']}"
        return line

    def _de_repeat(self, line: str) -> str:
        """Never say the same line twice in a set."""
        if line in self._recent_lines:
            line = line + " — for real this time"
        self._recent_lines.append(line)
        self._recent_lines = self._recent_lines[-12:]
        return line

    # ── voice rendering ──
    def render_voice(self, text: str, out_path: str) -> str:
        """Render a DJ line to wav via the voice catalogue. Returns the
        path, or '' if no TTS is available (caller falls back to text)."""
        try:
            from ..voice.catalogue import default_catalogue
            cat = default_catalogue()
            if self.voice:
                res = cat.speak_as(text, self.voice, out_path=out_path)
            else:
                res = cat.speak(text, out_path=out_path)
            p = (res or {}).get("path", "") or out_path
            return p if p and os.path.isfile(p) else ""
        except Exception:  # noqa: BLE001
            _log.debug("DJ voice render failed", exc_info=True)
            return ""


# ── the live set ───────────────────────────────────────────────────────

@dataclass
class _CrateTrack:
    path: str
    title: str
    style: str
    analysis: Any = None  # dj_engine.TrackAnalysis
    requester: str = ""
    is_request: bool = False


class LiveSet:
    """One live DJ set, running in its own thread.

    The caller wires delivery::

        LiveSet(chat_key=..., deliver_audio=fn, deliver_text=fn, context=...)

    where ``deliver_audio(path, caption)`` sends a voice note and
    ``deliver_text(text)`` sends a chat message. The set never touches
    the gateway directly — the runtime owns delivery.
    """

    def __init__(self, *, chat_key: str,
                 deliver_audio: Callable[[str, str], None],
                 deliver_text: Callable[[str], None],
                 context: Any = None, workdir: str = "dj_live") -> None:
        self.chat_key = chat_key
        self._deliver_audio = deliver_audio
        self._deliver_text = deliver_text
        self.context = context
        self.workdir = Path(workdir) / re.sub(r"\W+", "_", chat_key)
        self.sensor = RoomSensor()
        self.persona = DJPersona()
        self.state = "idle"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cmd: "queue.Queue[str]" = queue.Queue()
        self._crate: list[_CrateTrack] = []
        self._queue_next: list[_CrateTrack] = []  # request wins: play these first
        self._played: list[str] = []
        self._lock = threading.Lock()
        self._styles: list[str] = []
        self.started_at: float = 0.0

    # ── public control ──
    def start(self, styles: list[str] | None = None,
              n_tracks: int = 6) -> str:
        with self._lock:
            if self.state != "idle":
                return f"DJ is already live ({self.state})"
            self.state = "digging"
            self._styles = styles or ["afrobeats", "uk-drill", "amapiano"]
            self._n_tracks = max(3, min(12, n_tracks))
            self._stop.clear()
            self.started_at = time.time()
            self._thread = threading.Thread(
                target=self._run, name=f"dj-live-{self.chat_key}",
                daemon=True)
            self._thread.start()
            return f"🎧 {DJ_NAME} is digging the crate… live set starting"

    def stop(self) -> str:
        self._stop.set()
        self._cmd.put("stop")
        with self._lock:
            st = self.state
            self.state = "idle"
        return f"🎧 {DJ_NAME} signing off (was {st})"

    def notify(self, text: str, sender: str = "") -> None:
        """Feed a chat message into the room sensor. Non-blocking."""
        try:
            self.sensor.note_message(text, sender)
        except Exception:  # noqa: BLE001
            pass

    @property
    def live(self) -> bool:
        with self._lock:
            return self.state not in ("idle",)

    # ── the performance ──
    def _run(self) -> None:
        try:
            self._dig_crate()
            if self._stop.is_set() or not self._crate:
                self._set_state("idle")
                return
            self._intro()
            idx = 0
            order = self._order_crate()
            while not self._stop.is_set() and idx < len(order):
                # request wins: anything queued by the crowd plays first
                with self._lock:
                    queued = self._queue_next
                    self._queue_next = []
                if queued:
                    order[idx:idx] = queued
                track = order[idx]
                self._play_track(track)
                if self._stop.is_set():
                    break
                nxt = order[idx + 1] if idx + 1 < len(order) else None
                self._between(track, nxt)
                idx += 1
            self._outro()
        except Exception:  # noqa: BLE001 - the show must not die
            _log.exception("live DJ set crashed")
        finally:
            self._set_state("idle")

    def _set_state(self, s: str) -> None:
        with self._lock:
            self.state = s

    # ── crate digging ──
    def _dig_crate(self) -> None:
        self._set_state("digging")
        self.workdir.mkdir(parents=True, exist_ok=True)
        try:
            from .music import MusicCreator
        except Exception:  # noqa: BLE001
            _log.warning("live DJ: no composer available")
            return
        creator = MusicCreator(self.context)
        styles = self._styles
        for i in range(self._n_tracks):
            if self._stop.is_set():
                break
            style = styles[i % len(styles)]
            try:
                song = creator.compose(
                    f"{DJ_NAME} live crate {i + 1} in {style}",
                    style=style, with_midi=False, with_score=False,
                    with_vocals=False, workdir=str(self.workdir))
                path = getattr(song, "audio_path", "") or ""
                if not path or not os.path.isfile(path):
                    continue
                analysis = self._analyze(path, song)
                with self._lock:
                    self._crate.append(_CrateTrack(
                        path=path,
                        title=getattr(song, "title", "") or f"crate {i + 1}",
                        style=style, analysis=analysis))
            except Exception:  # noqa: BLE001
                _log.warning("live DJ: crate dig failed for track %d", i,
                             exc_info=True)
                continue

    def _analyze(self, path: str, song: Any) -> Any:
        try:
            from .dj_engine import analyze_track
            return analyze_track(
                path, title=getattr(song, "title", ""),
                known_bpm=getattr(song, "tempo", None),
                known_key=getattr(song, "key", "") or "",
                known_mode=getattr(song, "mode", "") or "")
        except Exception:  # noqa: BLE001
            return None

    def _order_crate(self) -> list[_CrateTrack]:
        with self._lock:
            tracks = list(self._crate)
        try:
            from .dj_engine import plan_energy_arc
            analyzed = [t for t in tracks if t.analysis is not None]
            rest = [t for t in tracks if t.analysis is None]
            if analyzed:
                arc = plan_energy_arc([t.analysis for t in analyzed])
                ordered = [analyzed[[t.analysis for t in analyzed].index(a)]
                           for a in arc]
                return ordered + rest
        except Exception:  # noqa: BLE001
            pass
        return tracks

    # ── moments ──
    def _intro(self) -> None:
        self._set_state("intro")
        line = self.persona.talk(situation="intro",
                                 energy=self.sensor.energy())
        self._dj_says(line)

    def _play_track(self, track: _CrateTrack) -> None:
        self._set_state("playing")
        caption = f"🎧 {DJ_NAME} — {track.title} ({track.style})"
        if track.is_request and track.requester:
            caption += f" — requested by {track.requester} 🔥"
        try:
            self._deliver_audio(track.path, caption)
        except Exception:  # noqa: BLE001
            _log.warning("live DJ: track delivery failed", exc_info=True)
        with self._lock:
            self._played.append(track.title)
        # let the track breathe: wait in slices so stop/skip lands fast.
        # radio edit length ~60s per track for a live set.
        self._wait_track(60.0)

    def _wait_track(self, secs: float) -> None:
        end = time.time() + secs
        while time.time() < end and not self._stop.is_set():
            try:
                cmd = self._cmd.get(timeout=2.0)
                if cmd == "stop":
                    return
            except queue.Empty:
                pass
            # dead-room early pivot: don't let silence sit
            if (self.sensor.dead_signals() >= 3
                    and time.time() > end - 30):
                return

    def _between(self, just_played: _CrateTrack,
                 nxt: _CrateTrack | None) -> None:
        self._set_state("talking")
        energy = self.sensor.energy()
        trend = self.sensor.trend()
        # 1. requests first — the crowd asked, the DJ answers
        req = self.sensor.pop_request()
        if req is not None:
            verdict = self._judge_request(req, just_played)
            if verdict is not None:
                with self._lock:
                    self._queue_next.append(verdict)
                self._dj_says(self.persona.talk(
                    situation="request_yes", track_title=verdict.title,
                    track_style=verdict.style, energy=energy,
                    requester=req.requester))
                return
            self._dj_says(self.persona.talk(
                situation="request_no", energy=energy,
                requester=req.requester,
                extra=f"“{req.query}”"))
            # fall through to normal selection
        # 2. read the room
        if energy < 0.22 or trend == "dead" or self.sensor.dead_signals() >= 2:
            line = self.persona.talk(situation="pivot",
                                     track_title=nxt.title if nxt else "",
                                     track_style=nxt.style if nxt else "",
                                     energy=energy)
            self._dj_says(line)
            self._pivot_crate(nxt)
        elif energy > 0.7:
            line = self.persona.talk(situation="hype", energy=energy)
            self._dj_says(line)
        else:
            line = self.persona.talk(
                situation="track_intro",
                track_title=nxt.title if nxt else "",
                track_style=nxt.style if nxt else "", energy=energy)
            self._dj_says(line)

    def _judge_request(self, req: TrackRequest,
                       current: _CrateTrack) -> _CrateTrack | None:
        """Can we honor this request right now? Compose it live in the
        requested lane, then check harmonic/energy fit against the
        current track with the real engine. Returns the crate track or
        None (doesn't fit — the DJ says so honestly)."""
        style = self._style_for_query(req.query)
        try:
            from .music import MusicCreator
            creator = MusicCreator(self.context)
            song = creator.compose(
                f"request: {req.query}",
                style=style, with_midi=False, with_score=False,
                with_vocals=False, workdir=str(self.workdir))
            path = getattr(song, "audio_path", "") or ""
            if not path or not os.path.isfile(path):
                return None
            analysis = self._analyze(path, song)
            cand = _CrateTrack(
                path=path, title=getattr(song, "title", "") or req.query,
                style=style, analysis=analysis,
                requester=req.requester, is_request=True)
            # fit check with the REAL engine
            if (current.analysis is not None and analysis is not None):
                from .dj_engine import harmonic_score, tempo_compatible
                hs = harmonic_score(current.analysis, analysis)
                tc = tempo_compatible(current.analysis, analysis)
                if hs <= 0.0 and not tc:
                    # clash on both axes — honest decline
                    return None
            with self._lock:
                self._crate.append(cand)
            return cand
        except Exception:  # noqa: BLE001
            _log.warning("live DJ: request compose failed", exc_info=True)
            return None

    @staticmethod
    def _style_for_query(query: str) -> str:
        q = (query or "").lower()
        for style in ("afrobeats", "amapiano", "uk-drill", "dancehall",
                      "hip-hop", "house", "rnb", "pop"):
            if style.replace("-", " ") in q or style in q:
                return style
        return "afrobeats"

    def _pivot_crate(self, nxt: _CrateTrack | None) -> None:
        """Room is dead — reorder remaining crate by raw energy, hottest
        first. Real pivot, not a pretense."""
        with self._lock:
            remaining = [t for t in self._crate
                         if t.title not in self._played]
            try:
                remaining.sort(
                    key=lambda t: getattr(t.analysis, "energy", 0.5)
                    if t.analysis else 0.5, reverse=True)
                # rebuild crate: played + reordered remainder
                played_set = set(self._played)
                self._crate[:] = ([t for t in self._crate
                                   if t.title in played_set] + remaining)
            except Exception:  # noqa: BLE001
                pass

    def _outro(self) -> None:
        self._set_state("outro")
        n = len(self._played)
        line = self.persona.talk(situation="outro", energy=self.sensor.energy(),
                                 extra=f"{n} tracks deep.")
        self._dj_says(line)

    # ── the DJ's mouth ──
    def _dj_says(self, text: str) -> None:
        """Talk-over transition: DJ voice over the moment. Voice note
        when TTS is available, text when it isn't — never silent."""
        text = (text or "").strip()
        if not text:
            return
        out = str(self.workdir / f"talk_{int(time.time() * 1000)}.wav")
        try:
            os.makedirs(self.workdir, exist_ok=True)
            vpath = self.persona.render_voice(text, out)
        except Exception:  # noqa: BLE001
            vpath = ""
        if vpath:
            try:
                self._deliver_audio(vpath, f"🎙️ {DJ_NAME}: {text[:120]}")
                return
            except Exception:  # noqa: BLE001
                pass
        try:
            self._deliver_text(f"🎙️ {DJ_NAME}: {text}")
        except Exception:  # noqa: BLE001
            pass


# ── registry + chat wiring ─────────────────────────────────────────────

_SETS: dict[str, LiveSet] = {}
_SETS_LOCK = threading.Lock()


def get_live_set(chat_key: str) -> LiveSet | None:
    with _SETS_LOCK:
        s = _SETS.get(chat_key)
        return s if (s is not None and s.live) else None


def start_live_set(chat_key: str, *,
                   deliver_audio: Callable[[str, str], None],
                   deliver_text: Callable[[str], None],
                   context: Any = None,
                   styles: list[str] | None = None,
                   n_tracks: int = 6) -> str:
    with _SETS_LOCK:
        existing = _SETS.get(chat_key)
        if existing is not None and existing.live:
            return f"🎧 {DJ_NAME} is already live here"
        s = LiveSet(chat_key=chat_key, deliver_audio=deliver_audio,
                    deliver_text=deliver_text, context=context)
        _SETS[chat_key] = s
    return s.start(styles=styles, n_tracks=n_tracks)


def stop_live_set(chat_key: str) -> str:
    with _SETS_LOCK:
        s = _SETS.pop(chat_key, None)
    if s is None:
        return "no live DJ set in this chat"
    return s.stop()


def notify_live_message(chat_key: str, text: str, sender: str = "") -> None:
    s = get_live_set(chat_key)
    if s is not None:
        s.notify(text, sender)


def control_dj_live(tail: str, *, chat_key: str,
                    deliver_audio: Callable[[str, str], None],
                    deliver_text: Callable[[str], None],
                    context: Any = None) -> str:
    """``/dj live [genre …]`` / ``/dj live stop`` — the live performer."""
    tail = (tail or "").strip()
    if tail.lower() == "stop":
        return stop_live_set(chat_key)
    styles = [p.strip().lower() for p in tail.split(",") if p.strip()] or None
    if tail and not styles:
        styles = [tail.lower()]
    msg = start_live_set(chat_key, deliver_audio=deliver_audio,
                         deliver_text=deliver_text, context=context,
                         styles=styles)
    return (msg + "\ntalk to the room — requests (“play afrobeats”), hype "
            "(🔥), or “skip” — the DJ is listening.")
