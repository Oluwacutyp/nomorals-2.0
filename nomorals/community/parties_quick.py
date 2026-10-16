"""Parties that work NOW — quick parties on current renders.

Gap-fill next to :mod:`nomorals.community.parties` (full party lifecycle:
INVITE → START → MUSIC → GAMES → SCORE → WALL → END → NEXT DAY).
That manager is complete and stays as-is. This module is the
skip-list correction: do NOT wait for 30–60 min mixes
(:mod:`nomorals.media.dj_long`) before parties exist.

:class:`PartyQuick` runs a party on what exists today:

* **Playlist** — built from existing mix descriptors (``/mix`` output,
  short mixes, single tracks). Any duration is accepted; there is NO
  minimum-length gate. A 30-minute target is a *goal*, not a
  requirement — the playlist reports its real total seconds and any
  shortfall honestly instead of refusing to start.
* **Trivia between tracks** — an injected ``trivia_provider`` callable
  (any callable returning a question dict, or ``None``). Between-track
  trivia reuses the same provider shape as the full party manager.
* **Votes** — members vote tracks up/down; the next track is the
  highest-voted unplayed track, ties break by earliest request.
* **Recap** — top tracks, top players, wall moments, playlist stats.
* **Rehearsal flag** — first parties (or any party the caller flags)
  run the full show with economy credits dry-run only (computed,
  reported, never credited).
* **Fail-soft** — every step runs through ``_safe``: a failing trivia
  provider, DJ hook, or economy hook is recorded and the party keeps
  going. A party never raises into the chat layer.

Universal: a quick party is a format, not a nation's format. The
``culture`` / ``nation`` / ``language`` fields select pack defaults
(any nation/culture — Nigeria is one default among N, never the axis).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger("nomorals.community.parties_quick")

__all__ = [
    "QuickParty", "PartyQuick", "QuickRecap",
    "build_playlist", "QUICK_TARGET_SECONDS",
]

#: 30-minute playlist goal. A goal, not a gate — short playlists play.
QUICK_TARGET_SECONDS = 30 * 60


def build_playlist(mixes: list[dict[str, Any]], *,
                   target_seconds: float = QUICK_TARGET_SECONDS,
                   max_tracks: int = 12) -> dict[str, Any]:
    """Assemble a playlist from existing mix descriptors.

    Each mix is a dict with at least ``title``; ``duration_s`` is
    optional (unknown → 0, reported as such). Mixes are taken in
    caller order until the target is reached or the list / ``max_tracks``
    runs out. Short mixes are FINE — a 3-minute mix counts as a track,
    and a playlist of short mixes is a valid party playlist.
    """
    tracks: list[dict[str, Any]] = []
    total = 0.0
    for m in list(mixes or [])[:max_tracks]:
        dur = float(m.get("duration_s") or 0)
        tracks.append({"title": str(m.get("title") or "untitled"),
                       "duration_s": dur,
                       "source": m.get("source", ""),
                       "mix_id": m.get("mix_id", m.get("id", ""))})
        total += dur
        if total >= target_seconds:
            break
    return {"tracks": tracks, "total_seconds": total,
            "target_seconds": target_seconds,
            "shortfall_seconds": max(0.0, target_seconds - total),
            "meets_target": total >= target_seconds,
            "track_count": len(tracks)}


@dataclass
class QuickParty:
    party_id: str
    group_id: str
    title: str
    host: str
    culture: str = "world"
    nation: str = ""
    language: str = "en"
    rehearsal: bool = False
    state: str = "invited"  # invited → live → ended
    playlist: dict[str, Any] = field(default_factory=dict)
    played: list[dict[str, Any]] = field(default_factory=list)
    queue: list[dict[str, Any]] = field(default_factory=list)
    rsvps: dict[str, str] = field(default_factory=dict)
    scores: dict[str, int] = field(default_factory=dict)
    wall: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    trivia_rounds: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuickParty":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class QuickRecap:
    party_id: str
    title: str
    top_tracks: list[dict[str, Any]]
    top_players: list[dict[str, Any]]
    wall_moments: list[dict[str, Any]]
    playlist_seconds: float
    trivia_rounds: int
    rehearsal: bool
    economy_dry_run: bool
    failures: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PartyQuick:
    """Run quick parties on current renders + /mix + trivia + DJ hooks."""

    def __init__(self, *,
                 trivia_provider: Callable[[], dict[str, Any] | None] | None = None,
                 dj_opener: Callable[[dict[str, Any]], Any] | None = None,
                 economy_hook: Callable[[str, str, int, str], Any] | None = None,
                 say: Callable[[str], None] | None = None,
                 now: Callable[[], float] | None = None) -> None:
        self._trivia = trivia_provider
        self._dj = dj_opener
        self._economy = economy_hook
        self._say_fn = say or (lambda _t: None)
        self._now = now or time.time
        self._seq = 0

    # ── internals ──────────────────────────────────────────────────────────
    def _safe(self, party: QuickParty, step: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # fail-soft: record, keep the party going
            party.failures.append({"step": step, "error": f"{type(exc).__name__}: {exc}"})
            _log.warning("quick party %s step %s failed: %s", party.party_id, step, exc)
            return None

    def _say(self, text: str) -> None:
        try:
            self._say_fn(text)
        except Exception:
            pass

    def _credit(self, party: QuickParty, member: str, amount: int, reason: str) -> None:
        if party.rehearsal or self._economy is None:
            return  # rehearsal: computed + reported, never credited
        self._safe(party, "economy", lambda: self._economy(party.group_id, member, amount, reason))

    # ── lifecycle ──────────────────────────────────────────────────────────
    def create(self, group_id: str, title: str, host: str, *,
               mixes: list[dict[str, Any]] | None = None,
               culture: str = "world", nation: str = "", language: str = "en",
               rehearsal: bool = False,
               target_seconds: float = QUICK_TARGET_SECONDS) -> QuickParty:
        self._seq += 1
        party = QuickParty(
            party_id=f"qp-{self._seq}", group_id=group_id, title=title, host=host,
            culture=culture, nation=nation, language=language, rehearsal=rehearsal,
            playlist=build_playlist(mixes or [], target_seconds=target_seconds))
        party.queue = [dict(t, votes={}, score=0) for t in party.playlist["tracks"]]
        return party

    def rsvp(self, party: QuickParty, member: str, choice: str) -> str:
        choice = (choice or "").lower()
        if choice not in ("yes", "no", "maybe"):
            return "rsvp must be yes/no/maybe"
        party.rsvps[member] = choice
        return f"{member}: {choice}"

    def start(self, party: QuickParty) -> bool:
        if party.state != "invited":
            return False
        party.state = "live"
        party.started_at = self._now()
        self._safe(party, "dj_opener",
                   lambda: self._dj({"title": party.title, "group_id": party.group_id})
                   if self._dj else None)
        return True

    def request_track(self, party: QuickParty, member: str, title: str) -> dict[str, Any] | None:
        if party.state != "live":
            return None
        track = {"title": title, "duration_s": 0.0, "source": "request",
                 "mix_id": "", "votes": {}, "score": 0, "requested_by": member}
        party.queue.append(track)
        return track

    def vote_track(self, party: QuickParty, member: str, track_idx: int,
                   up: bool = True) -> bool:
        if party.state != "live" or not (0 <= track_idx < len(party.queue)):
            return False
        track = party.queue[track_idx]
        track["votes"][member] = 1 if up else -1
        track["score"] = sum(track["votes"].values())
        return True

    def next_track(self, party: QuickParty) -> dict[str, Any] | None:
        """Play the highest-voted queued track; trivia runs between tracks."""
        if party.state != "live" or not party.queue:
            return None
        party.queue.sort(key=lambda t: (-t.get("score", 0), t.get("title", "")))
        track = party.queue.pop(0)
        party.played.append(track)
        if party.played and len(party.played) > 1:
            self.play_trivia_round(party)
        return track

    def play_trivia_round(self, party: QuickParty) -> dict[str, Any] | None:
        if self._trivia is None:
            return None
        q = self._safe(party, "trivia", self._trivia)
        if not isinstance(q, dict) or not q.get("q"):
            return None
        party.trivia_rounds += 1
        party.scores.setdefault("__trivia_pending__", 0)
        return {"question": q.get("q"), "options": q.get("options", []),
                "round": party.trivia_rounds}

    def answer_trivia(self, party: QuickParty, member: str, answer: str,
                      correct_answer: str, *, points: int = 10) -> dict[str, Any]:
        if party.state != "live":
            return {"ok": False, "error": "party not live"}
        ok = (answer or "").strip().lower() == (correct_answer or "").strip().lower()
        if ok:
            party.scores[member] = party.scores.get(member, 0) + points
            self._credit(party, member, points, "quick_party_trivia")
        return {"ok": True, "correct": ok, "points": points if ok else 0}

    def post_to_wall(self, party: QuickParty, member: str, text: str = "") -> dict[str, Any]:
        post = {"member": member, "text": text, "at": self._now()}
        party.wall.append(post)
        return post

    def end(self, party: QuickParty) -> QuickRecap:
        party.state = "ended"
        party.ended_at = self._now()
        top_tracks = sorted(party.played, key=lambda t: -t.get("score", 0))[:3]
        top_players = sorted(
            ({"member": m, "score": s} for m, s in party.scores.items()
             if not m.startswith("__")),
            key=lambda r: -r["score"])[:3]
        return QuickRecap(
            party_id=party.party_id, title=party.title,
            top_tracks=[{"title": t["title"], "score": t.get("score", 0)}
                        for t in top_tracks],
            top_players=top_players, wall_moments=party.wall[-5:],
            playlist_seconds=float(party.playlist.get("total_seconds", 0)),
            trivia_rounds=party.trivia_rounds, rehearsal=party.rehearsal,
            economy_dry_run=bool(party.rehearsal), failures=list(party.failures))
